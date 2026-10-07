"""Write complete run configs for another cache/device without editing tracked files."""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--config-output", type=Path, default=Path(".local/portable_configs"))
    parser.add_argument("--results-output", type=Path, default=Path("results/portable"))
    parser.add_argument("--devices", nargs=3, default=["cuda:0", "cuda:1", "cuda:2"])
    parser.add_argument(
        "--image-root", type=Path, help="Prepared CIFAR cache, default DATA/cifar10"
    )
    args = parser.parse_args()
    cache = args.data_root.resolve()
    args.config_output.mkdir(parents=True, exist_ok=True)
    for path in sorted((ROOT / "configs").glob("*.json")):
        config = json.loads(path.read_text())
        if "source_config" in config:
            config["source_config"] = str(args.config_output / Path(config["source_config"]).name)
        if "data_root" in config:
            config["data_root"] = str(cache)
        for key in ["pretrained_weights", "classifier_weights"]:
            if config.get(key):
                config[key] = str(cache / "models/torch/hub/checkpoints" / Path(config[key]).name)
        if "image_root" in config:
            config["image_root"] = str((args.image_root or cache / "cifar10").resolve())
        if "source_root" in config:
            config["source_root"] = str(cache / "models/DDColor")
            config["model_root"] = str(cache / "models/ddcolor_modelscope")
        if "mean_checkpoint" in config:
            config["mean_checkpoint"] = str(
                args.results_output / "generative_colorization/l1/checkpoints/l1_0.pt"
            )
        if "baseline_root" in config:
            config["baseline_root"] = str(args.results_output)
        if "warm_start" in config:
            config["warm_start"] = str(
                args.results_output / "generative_{task}/l1/checkpoints/l1_{seed}.pt"
            )
        if "device" in config:
            index = (
                int(config["device"].split(":")[-1]) if config["device"].startswith("cuda:") else 0
            )
            config["device"] = args.devices[index]
        output_name = (
            "generative_colorization/cvae"
            if path.stem == "generative_colorization_cvae"
            else path.stem
        )
        config["output"] = str(args.results_output / output_name)
        (args.config_output / path.name).write_text(json.dumps(config, indent=2) + "\n")
    print("Configs written to", args.config_output)


if __name__ == "__main__":
    main()
