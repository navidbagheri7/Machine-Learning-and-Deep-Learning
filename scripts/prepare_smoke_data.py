"""Make offline procedural fixtures for CI integration, never scientific reference data."""

import argparse
import json
from pathlib import Path

import numpy as np


def prepare(root):
    directory = root / "projects"
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(13)
    coords = np.mgrid[-1:1:28j, -1:1:28j]

    def digits(n):
        images = []
        for _ in range(n):
            center = rng.uniform(-0.5, 0.5, 2)
            image = np.exp(-((coords[0] - center[0]) ** 2 + (coords[1] - center[1]) ** 2) / 0.08)
            images.append((image * 255).astype(np.uint8))
        return np.stack(images)

    np.savez_compressed(directory / "mnist.npz", train=digits(600), test=digits(64))
    axis = np.mgrid[0:1:96j, 0:1:96j]

    def photos(n):
        images = []
        for _ in range(n):
            color = rng.uniform(0.1, 0.9, 3)
            texture = np.cos(axis[0] * rng.uniform(5, 20)) * np.sin(axis[1] * rng.uniform(5, 20))
            images.append(((0.5 + texture[..., None] * color / 2) * 255).astype(np.uint8))
        return np.stack(images)

    np.savez_compressed(directory / "coco96.npz", train=photos(256), val=photos(128))
    for number in [1, 2]:
        x = np.concatenate(
            [rng.normal(size=(200, 2)) + center for center in [[-3, -2], [0, 3], [3, -2]]]
        )
        np.save(directory / f"mixture_{number}.npy", x)
    (directory / "audit.json").write_text(
        json.dumps({"archives": [], "source": "Procedural CI fixtures; not MNIST or COCO evidence"})
        + "\n"
    )
    print("Prepared offline integration fixtures", directory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(".local/smoke_data"))
    args = parser.parse_args()
    prepare(args.data_root)
    config = json.loads(Path("configs/colorization.json").read_text())
    config["pretrained_weights"] = None
    config["data_root"] = str(args.data_root)
    (args.data_root / "colorization.json").write_text(json.dumps(config) + "\n")
