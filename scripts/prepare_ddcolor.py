"""Explicitly prepare the pinned Apache-2.0 DDColor source and official model weights."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=Path("/data/navid.bagheri-shouraki/ML_data"))
    args = p.parse_args()
    config = json.loads((ROOT / "configs" / "ddcolor.json").read_text())
    source = args.data_root / "models" / "DDColor"
    source.parent.mkdir(parents=True, exist_ok=True)
    if not source.exists():
        subprocess.run(
            ["git", "clone", "https://github.com/piddnad/DDColor.git", str(source)], check=True
        )
    current = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"], check=True, text=True, capture_output=True
    ).stdout.strip()
    if current != config["source_revision"]:
        subprocess.run(
            ["git", "-C", str(source), "checkout", "--detach", config["source_revision"]],
            check=True,
        )
    model = args.data_root / "models" / "ddcolor_modelscope"
    snapshot_download(
        "piddnad/ddcolor_modelscope",
        revision=config["model_revision"],
        allow_patterns=["config.json", "pytorch_model.bin", "README.md"],
        local_dir=model,
        cache_dir=args.data_root / "models" / "huggingface",
    )
    if (
        hashlib.sha256((model / "pytorch_model.bin").read_bytes()).hexdigest()
        != config["weight_sha256"]
    ):
        raise ValueError("Pinned DDColor weight checksum mismatch")
    manifest = json.loads((ROOT / "assets/pretrained_manifest.json").read_text())
    if (
        hashlib.sha256((model / "config.json").read_bytes()).hexdigest()
        != manifest["model_config_sha256"]
    ):
        raise ValueError("Pinned DDColor model configuration checksum mismatch")
    config.update(data_root=str(args.data_root), source_root=str(source), model_root=str(model))
    destination = args.data_root / "models" / "ddcolor_config.json"
    destination.write_text(json.dumps(config, indent=2) + "\n")
    print("Prepared pinned DDColor; use --config", destination)
