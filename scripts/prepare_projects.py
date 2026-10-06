"""Audit private coursework and cache deterministic, attributed image benchmarks.

Run with --data-root pointing to a writable personal cache. No shared data is modified.
The private handouts are never copied into the public repository.
"""

import argparse
import concurrent.futures
import hashlib
import io
import json
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def retrieve(url, path):
    # COCO's dotted image hostname does not match the storage TLS certificate.
    # Use the same official S3 bucket through its correctly certificated path endpoint.
    if url.startswith(("http://images.cocodataset.org/", "https://images.cocodataset.org/")):
        url = (
            "https://s3.amazonaws.com/images.cocodataset.org/"
            + url.split("images.cocodataset.org/", 1)[1]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    for attempt in range(3):
        try:
            with (
                urllib.request.urlopen(url.replace("http://", "https://"), timeout=60) as response,
                temporary.open("wb") as handle,
            ):
                while chunk := response.read(1024 * 1024):
                    handle.write(chunk)
            temporary.replace(path)
            return
        except OSError:
            temporary.unlink(missing_ok=True)
            if attempt == 2:
                raise


def download_sources(shared):
    """Fetch official MNIST and COCO metadata; image downloads are selected later."""
    from torchvision.datasets import MNIST

    MNIST(shared / "mnist" / "raw", train=True, download=True)
    MNIST(shared / "mnist" / "raw", train=False, download=True)
    coco = shared / "coco2017"
    annotation_dir = coco / "annotations"
    if all((annotation_dir / f"captions_{split}2017.json").exists() for split in ["train", "val"]):
        return
    archive = coco / "archives" / "annotations_trainval2017.zip"
    if not archive.exists():
        retrieve("https://images.cocodataset.org/annotations/annotations_trainval2017.zip", archive)
    manifest = json.loads((ROOT / "assets" / "project_data_manifest.json").read_text())
    if sha(archive) != manifest["coco_annotation_archive_sha256"]:
        raise ValueError("Official COCO annotation archive checksum mismatch")
    annotation_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        # Never extract arbitrary archive paths.
        for split in ["train", "val"]:
            name = f"captions_{split}2017.json"
            (annotation_dir / name).write_bytes(z.read("annotations/" + name))


def prepare(data_root, shared, originals, train_count=8000, heldout_count=800, download=False):
    cache = Path(data_root) / "projects"
    cache.mkdir(parents=True, exist_ok=True)
    audit = {"sources": [], "archives": [], "natural_images": {}}
    for path in sorted(originals.iterdir()) if originals.exists() else []:
        audit["sources"].append(
            {"name": path.name, "sha256": sha(path), "bytes": path.stat().st_size}
        )
    for i, path in enumerate(sorted(originals.glob("Image*.csv")), 1):
        points = np.loadtxt(path, delimiter=",", skiprows=1)[:, -2:]
        np.save(cache / f"mixture_{i}.npy", points)
    if not list(originals.glob("Image*.csv")):
        points = np.load(ROOT / "assets" / "source_data" / "assignment_points.npz")
        for i in [1, 2]:
            np.save(cache / f"mixture_{i}.npy", points[f"image{i}"])
    for path in sorted(originals.glob("*.zip")):
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            clean = {Path(n).stem: n for n in names if "/original/" in n and n.endswith(".png")}
            noisy = {Path(n).stem: n for n in names if "/corrupted/" in n and n.endswith(".png")}
            ids = sorted(clean.keys() & noisy.keys(), key=int)
            matrix = np.load(io.BytesIO(archive.read("MNIST/W.npy")), allow_pickle=False)
            m = int(np.sqrt(matrix.shape[0]))
            assert matrix.shape == (m * m, m * m)

            def images(mapping):
                return np.stack(
                    [np.asarray(Image.open(io.BytesIO(archive.read(mapping[k])))) for k in ids]
                )

            np.savez_compressed(
                cache / f"course_m{m}.npz",
                clean=images(clean),
                observed=images(noisy),
                operator=matrix,
                ids=np.array(ids, int),
            )
            audit["archives"].append(
                {
                    "file": path.name,
                    "actual_patch_size": m,
                    "operator_shape": list(matrix.shape),
                    "paired_count": len(ids),
                    "unpaired_original_ids": sorted(clean.keys() - noisy.keys(), key=int),
                    "unpaired_corrupted_ids": sorted(noisy.keys() - clean.keys(), key=int),
                }
            )
    # Read raw IDX files directly: downloading MNIST is unnecessary on this host.
    mnist = shared / "mnist" / "raw" / "MNIST" / "raw"
    train = np.frombuffer(
        (mnist / "train-images-idx3-ubyte").read_bytes(), np.uint8, offset=16
    ).reshape(-1, 28, 28)
    test = np.frombuffer(
        (mnist / "t10k-images-idx3-ubyte").read_bytes(), np.uint8, offset=16
    ).reshape(-1, 28, 28)
    np.savez_compressed(cache / "mnist.npz", train=train, test=test)
    # Pin exact COCO image IDs and URLs; publish gallery photos only with attribution.
    rng = np.random.default_rng(42)
    coco = shared / "coco2017"
    images = {}
    for split, count in [("train", train_count), ("val", heldout_count)]:
        metadata = json.loads((coco / "annotations" / f"captions_{split}2017.json").read_text())
        eligible = sorted(
            [x for x in metadata["images"] if x["license"] == 4], key=lambda x: x["id"]
        )
        order = rng.permutation(len(eligible))[:count]
        chosen = [eligible[j] for j in order]
        if len(chosen) != count:
            raise ValueError("Insufficient CC BY images")
        if download:

            def fetch(x):
                path = coco / f"{split}2017" / x["file_name"]
                if not path.exists():
                    retrieve(x["coco_url"], path)

            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                list(pool.map(fetch, chosen))
        pixels = []
        for x in chosen:
            with Image.open(coco / f"{split}2017" / x["file_name"]) as image:
                pixels.append(
                    np.asarray(
                        ImageOps.fit(image.convert("RGB"), (96, 96), Image.Resampling.BICUBIC)
                    )
                )
        images[split] = np.stack(pixels)
        audit["natural_images"][split] = [
            {
                "id": x["id"],
                "file": x["file_name"],
                "flickr_url": x["flickr_url"],
                "license": "CC BY 2.0",
            }
            for x in chosen
        ]
    np.savez_compressed(cache / "coco96.npz", **images)
    audit["cache_sha256"] = {p.name: sha(p) for p in sorted(cache.glob("*.npz"))}
    audit["natural_images"]["protocol"] = (
        f"{train_count} train2017; first {heldout_count // 2} selected val2017 validation, remaining images test; CC BY 2.0 only; 96px center crop"
    )
    (cache / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print("Cached project data in", cache)
    for row in audit["archives"]:
        print(row["file"], "actual m:", row["actual_patch_size"], "pairs:", row["paired_count"])
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", type=Path, default=Path("/data/navid.bagheri-shouraki/ML_data")
    )
    parser.add_argument("--shared", type=Path, default=Path("/data/tii/Image_data"))
    parser.add_argument(
        "--originals", type=Path, default=ROOT / "legacy" / "originals" / "projects"
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download official data into DATA_ROOT/project_sources instead of using --shared",
    )
    args = parser.parse_args()
    shared = args.data_root / "project_sources" if args.download else args.shared
    if args.download:
        download_sources(shared)
    prepare(args.data_root, shared, args.originals, download=args.download)
