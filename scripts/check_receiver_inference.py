"""Reproduce every saved neural-receiver benchmark condition and exercise its CLI."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from learning_lab.artifacts import write_json  # noqa: E402
from learning_lab.receiver_inference import receive  # noqa: E402
from learning_lab.signals import constellation, generate  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=ROOT / "results/reference/signals")
    parser.add_argument("--device", help="Default is the reference record's original device")
    parser.add_argument("--compare-cpu", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT / ".local/receiver_inference")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    record = json.loads((args.results_root / "run.json").read_text())
    assert record["status"] == "complete" and not record["config"].get("quick")
    config = record["config"]
    args.device = args.device or config["device"]
    if args.compare_cpu and not args.device.startswith("cuda"):
        parser.error("CPU comparison requires a CUDA reference device")
    reference = json.loads((args.results_root / "metrics.json").read_text())["rows"]
    _, bits = constellation()
    checks, command_checks, cpu_numerics = [], [], []
    environment = os.environ | {"OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4"}
    for seed in config["seeds"]:
        for kind in ["mlp", "cnn"]:
            checkpoint = args.results_root / f"checkpoints/{kind}_{seed}.pt"
            for gamma in [0.2, 0.4]:
                for snr in [10, 14, 18, 22, 26]:
                    observed, truth, _ = generate(
                        config["test_blocks"], 64, snr, gamma, 9000 + seed
                    )
                    prediction = receive(checkpoint, observed, args.device)
                    symbols = prediction["symbol_classes"]
                    ber = float(np.mean(prediction["gray_bits"] != bits[truth]))
                    ser = float(np.mean(symbols != truth))
                    expected = next(
                        r
                        for r in reference
                        if (r["model"], r["seed"], r["gamma"], r["snr_db"])
                        == (kind, seed, gamma, snr)
                    )
                    assert abs(ber - expected["ber"]) < 1e-12, (kind, seed, gamma, snr, "BER")
                    assert abs(ser - expected["ser"]) < 1e-12, (kind, seed, gamma, snr, "SER")
                    if args.compare_cpu:
                        cpu = receive(checkpoint, observed, "cpu")
                        disagreement = float(np.mean(cpu["symbol_classes"] != symbols))
                        probability_difference = float(
                            np.max(np.abs(cpu["probabilities"] - prediction["probabilities"]))
                        )
                        # Numerical acceptance thresholds fixed before the full comparison;
                        # actual differences are retained, never rounded into equivalence.
                        assert disagreement <= 1e-3 and probability_difference <= 0.02
                        cpu_numerics.append(
                            {
                                "model": kind,
                                "seed": seed,
                                "gamma": gamma,
                                "snr_db": snr,
                                "symbols": int(truth.size),
                                "different_symbol_classes": int(
                                    np.sum(cpu["symbol_classes"] != symbols)
                                ),
                                "maximum_probability_difference": probability_difference,
                                "cpu_ber": float(np.mean(cpu["gray_bits"] != bits[truth])),
                                "cuda_ber": ber,
                            }
                        )
                    checks.append(
                        {
                            "model": kind,
                            "seed": seed,
                            "gamma": gamma,
                            "snr_db": snr,
                            "blocks": len(truth),
                            "bits": int(truth.size * 4),
                            "ber_absolute_difference": abs(ber - expected["ber"]),
                            "ser_absolute_difference": abs(ser - expected["ser"]),
                        }
                    )
            source = args.output_dir / f"observed_{kind}_{seed}.npy"
            np.save(source, observed[:8])
            output = source.with_suffix(".npz")
            with source.with_suffix(".log").open("w") as log:
                subprocess.run(
                    [
                        sys.executable,
                        "scripts/predict_receiver.py",
                        "--checkpoint",
                        str(checkpoint.resolve()),
                        "--input",
                        str(source.resolve()),
                        "--output",
                        str(output.resolve()),
                        "--device",
                        args.device,
                    ],
                    cwd=ROOT,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            expected_subset = receive(checkpoint, observed[:8], args.device)
            with np.load(output) as deployed:
                for name in expected_subset:
                    np.testing.assert_array_equal(deployed[name], expected_subset[name])
            metadata = json.loads(output.with_suffix(".json").read_text())
            assert (
                metadata["checkpoint_sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            )
            assert metadata["input_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
            command_checks.append(
                {
                    "model": kind,
                    "seed": seed,
                    "checkpoint_sha256": metadata["checkpoint_sha256"],
                    "probabilities_match_api_exactly": True,
                    "device": args.device,
                }
            )
            if args.compare_cpu:
                cpu_output = source.with_name(source.stem + "_cpu.npz")
                with cpu_output.with_suffix(".log").open("w") as log:
                    subprocess.run(
                        [
                            sys.executable,
                            "scripts/predict_receiver.py",
                            "--checkpoint",
                            str(checkpoint.resolve()),
                            "--input",
                            str(source.resolve()),
                            "--output",
                            str(cpu_output.resolve()),
                        ],
                        cwd=ROOT,
                        env=environment,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
                expected_cpu = receive(checkpoint, observed[:8], "cpu")
                with np.load(cpu_output) as deployed:
                    for name in expected_cpu:
                        np.testing.assert_array_equal(deployed[name], expected_cpu[name])
                command_checks.append(
                    {
                        "model": kind,
                        "seed": seed,
                        "device": "cpu",
                        "probabilities_match_api_exactly": True,
                    }
                )
            print("Verified all receiver conditions and CLI", kind, seed, flush=True)
    write_json(
        args.output_dir / "audit.json",
        {
            "status": "passed",
            "device": args.device,
            "checks": checks,
            "commands": command_checks,
            "cpu_numerics": cpu_numerics,
            "numerical_settings": {
                "cuda_convolution_tf32_allowed": True,
                "cuda_matmul_tf32_allowed": False,
                "cpu_max_symbol_disagreement_fraction": 1e-3,
                "cpu_max_probability_difference": 0.02,
            },
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "source_sha256": {
                name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                for name in [
                    "scripts/predict_receiver.py",
                    "src/learning_lab/receiver_inference.py",
                    "src/learning_lab/signals.py",
                ]
            },
            "protocol": "Every neural benchmark row: MLP/CNN, three fitting seeds, two nonlinearities and five SNRs, with the original 2,000 independent blocks and generation seeds. Primary CUDA predictions use deterministic convolution with TF32 allowed and matrix TF32 disabled, matching the original runtime defaults; integer errors reproduce BER/SER to1e-12. Optional CPU comparison retains every class/probability/BER difference under declared acceptance bounds, rather than claiming bitwise cross-device equivalence. Fresh-process commands reproduce their device-specific API outputs exactly on8-block subsets. Neither targets nor channel settings enter inference. Probabilities are uncalibrated; no physical-link performance claim.",
        },
    )


if __name__ == "__main__":
    main()
