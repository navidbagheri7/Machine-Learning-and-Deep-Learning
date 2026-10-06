"""Explicitly prepare official CIFAR and ImageNet ResNet18 assets for a portable run."""

import argparse
import hashlib
import json
import tarfile
import urllib.request
from pathlib import Path


def download(url, path):
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(path.suffix + ".partial")
        with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as target:
            while block := response.read(1024 * 1024):
                target.write(block)
        partial.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=".local/cifar10")
    parser.add_argument("--weights", default=".local/models/resnet18-f37072fd.pth")
    args = parser.parse_args()
    root = Path(args.data_root)
    archive = root / "cifar-10-python.tar.gz"
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    download(url, archive)
    if hashlib.md5(archive.read_bytes()).hexdigest() != "c58f30108f718f92721af3b95e74349a":
        raise ValueError("CIFAR archive checksum mismatch")
    folder = root / "cifar-10-batches-py"
    folder.mkdir(parents=True, exist_ok=True)
    # Extract only the fixed expected files, with no paths from untrusted members.
    with tarfile.open(archive) as tar:
        for name in ["batches.meta", "test_batch", *[f"data_batch_{i}" for i in range(1, 6)]]:
            member = tar.extractfile("cifar-10-batches-py/" + name)
            if member is None:
                raise ValueError("Missing CIFAR member: " + name)
            (folder / name).write_bytes(member.read())
    weight_url = "https://download.pytorch.org/models/resnet18-f37072fd.pth"
    download(weight_url, args.weights)
    digest = hashlib.sha256(Path(args.weights).read_bytes()).hexdigest()
    if not digest.startswith("f37072fd"):
        raise ValueError("Pretrained weight checksum mismatch")
    manifest = {
        "cifar_url": url,
        "cifar_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "pretrained_url": weight_url,
        "pretrained_sha256": digest,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
