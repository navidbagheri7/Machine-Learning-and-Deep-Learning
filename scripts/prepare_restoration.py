"""Prepare a disjoint, higher-resolution CC BY COCO benchmark without touching shared data."""

import argparse
import concurrent.futures
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from prepare_projects import retrieve


def annotation_file(root, shared, source, download):
    name = f"captions_{source}2017.json"
    path = shared / "annotations" / name
    if path.exists():
        return path
    directory = root / "project_sources" / "coco2017"
    path = directory / "annotations" / name
    if path.exists():
        return path
    if not download:
        raise FileNotFoundError("COCO annotations missing; prepare with --download")
    archive = directory / "archives" / "annotations_trainval2017.zip"
    if not archive.exists():
        retrieve("https://images.cocodataset.org/annotations/annotations_trainval2017.zip", archive)
    expected = json.loads(
        (Path(__file__).resolve().parents[1] / "assets" / "project_data_manifest.json").read_text()
    )["coco_annotation_archive_sha256"]
    digest = hashlib.sha256()
    with archive.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise ValueError("Official COCO annotation archive checksum mismatch")
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        path.write_bytes(z.read("annotations/" + name))
    return path


def prepare(root, shared, count=16000, size=256, download=False):
    output = root / "restoration256"
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    record = {"size": size, "training_crop": 128, "license": "CC BY 2.0", "splits": {}}
    for source, budget in [("train", count), ("val", 800)]:
        annotation = annotation_file(root, shared, source, download)
        metadata = json.loads(annotation.read_text())
        eligible = sorted(
            [x for x in metadata["images"] if x["license"] == 4], key=lambda x: x["id"]
        )
        chosen = [eligible[j] for j in rng.permutation(len(eligible))[:budget]]
        if len(chosen) != budget:
            raise ValueError("Requested budget exceeds eligible images")
        temporary = output / f"{source}.npy.part"
        pixels = np.lib.format.open_memmap(
            temporary, mode="w+", dtype=np.uint8, shape=(budget, size, size, 3)
        )

        def convert(item, storage=pixels):
            i, row = item
            path = shared / f"{source}2017" / row["file_name"]
            if not path.exists():
                path = root / "project_sources" / "coco2017" / f"{source}2017" / row["file_name"]
            if not path.exists():
                if not download:
                    raise FileNotFoundError(f"{path}; use --download for missing photos")
                retrieve(row["coco_url"], path)
            with Image.open(path) as image:
                storage[i] = np.asarray(
                    ImageOps.fit(image.convert("RGB"), (size, size), Image.Resampling.BICUBIC)
                )
            return {
                "id": row["id"],
                "file": row["file_name"],
                "url": row["coco_url"],
                "flickr_url": row["flickr_url"],
                "original_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }

        with concurrent.futures.ThreadPoolExecutor(8) as executor:
            rows = list(executor.map(convert, enumerate(chosen)))
        pixels.flush()
        del convert
        del pixels
        temporary.replace(output / f"{source}.npy")
        if source == "train":
            record["splits"]["train"] = rows
        else:
            record["splits"]["validation"], record["splits"]["test"] = rows[:400], rows[400:]
        digest = hashlib.sha256()
        with (output / f"{source}.npy").open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        record[f"{source}_sha256"] = digest.hexdigest()
        print(source, len(rows), "prepared", flush=True)
    record["protocol"] = (
        "CC BY-only COCO2017; 16000 train2017; disjoint 400 validation and 400 test val2017; deterministic center-resized 256px images; training random 128px crops. No test selection."
    )
    (output / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", type=Path, default=Path("/data/navid.bagheri-shouraki/ML_data")
    )
    parser.add_argument("--shared", type=Path, default=Path("/data/tii/Image_data/coco2017"))
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    prepare(args.data_root, args.shared, download=args.download)
