"""Compare interrupted and uninterrupted training on prepared real images.

This is a CPU integration check, not a quality benchmark. It deliberately stops
after the first completed epoch, then verifies the final weights and losses.
"""

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import learning_lab.generative as generative  # noqa: E402
from learning_lab.artifacts import write_json  # noqa: E402


class ControlledStop(Exception):
    """Expected interruption after saving an epoch's complete resume state."""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(".local/resume_check"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    base = json.loads((ROOT / "configs/generative_super_resolution.json").read_text())
    base.update(
        quick=True,
        device="cpu",
        data_root=str(args.data_root),
        pretrained_weights=str(
            args.data_root / "models/torch/hub/checkpoints/resnet18-f37072fd.pth"
        ),
        train_images=16,
        validation_images=8,
        test_images=8,
        crop=32,
        width=16,
        latent=8,
        batch_size=8,
        epochs=2,
        seeds=[0],
        methods=["cvae"],
    )
    torch.hub.set_dir(str(args.data_root / "models/torch/hub"))
    args.output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="fresh_", dir=args.output) as temporary:
        uninterrupted_config = base | {"output": str(Path(temporary) / "uninterrupted")}
        generative.experiment(uninterrupted_config)
        expected = torch.load(
            Path(uninterrupted_config["output"]) / "checkpoints/cvae_0.pt",
            map_location="cpu",
            weights_only=True,
        )
        curve = json.loads((Path(uninterrupted_config["output"]) / "metrics.json").read_text())[
            "curves"
        ][0]
        resumed_config = base | {"output": str(Path(temporary) / "resumed")}
        save = generative.save_checkpoint

        def interrupt(value, path):
            save(value, path)
            if str(path).endswith("_training.pt") and len(value["history"]) == 1:
                raise ControlledStop()

        generative.save_checkpoint = interrupt
        stopped = False
        try:
            generative.experiment(resumed_config)
        except ControlledStop:
            stopped = True
        finally:
            generative.save_checkpoint = save
        assert stopped, "The deliberate interruption did not occur"
        record_path = Path(resumed_config["output"]) / "run.json"
        stopped_record = json.loads(record_path.read_text())
        assert stopped_record["status"] == "running"
        generative.experiment(resumed_config)
        final_record = json.loads(record_path.read_text())
        assert final_record["status"] == "complete"
        assert final_record["prior_attempts"][-1] == {
            k: v for k, v in stopped_record.items() if k not in ["config", "prior_attempts"]
        }
        resumed = torch.load(
            Path(resumed_config["output"]) / "checkpoints/cvae_0.pt",
            map_location="cpu",
            weights_only=True,
        )
        actual = json.loads((Path(resumed_config["output"]) / "metrics.json").read_text())[
            "curves"
        ][0]
        assert expected["best_epoch"] == resumed["best_epoch"]
        assert all(
            torch.equal(expected["state_dict"][k], v) for k, v in resumed["state_dict"].items()
        )
        assert len(curve["history"]) == len(actual["history"]) == 2
        for first, second in zip(curve["history"], actual["history"]):
            assert first["train"] == second["train"]
            assert first["validation_loss"] == second["validation_loss"]
        write_json(
            args.output / "audit.json",
            {
                "status": "passed",
                "method": "SR CVAE",
                "epochs": 2,
                "checkpoint_and_losses": "bitwise identical after interruption at epoch 1",
                "device": "cpu",
                "data": "16 real training images, 8 validation images; integration only",
                "verified_state_tensors": len(expected["state_dict"]),
                "prior_attempt_preserved": True,
                "completed_record_and_selected_epoch_verified": True,
                "training_source_sha256": hashlib.sha256(
                    (ROOT / "src/learning_lab/generative.py").read_bytes()
                ).hexdigest(),
                "audit_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            },
        )
    print("Interrupted/resumed training exactly matches uninterrupted fitting.")


if __name__ == "__main__":
    main()
