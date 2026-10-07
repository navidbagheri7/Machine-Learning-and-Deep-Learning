"""Verify prepared benchmark bytes and describe actual original-photo resolution."""

import argparse
import concurrent.futures
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from learning_lab.artifacts import write_json  # noqa: E402


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--shared", type=Path, default=Path("/data/tii/Image_data/coco2017"))
    parser.add_argument("--output", type=Path, default=Path(".local/restoration_data_audit.json"))
    args = parser.parse_args()
    root = args.data_root / "restoration256"
    manifest = json.loads((root / "manifest.json").read_text())
    published = json.loads((ROOT / "assets/restoration_data_manifest.json").read_text())
    assert manifest == published, "Cache manifest differs from published benchmark"
    checks = []
    for name, count in [("train", 16000), ("val", 800)]:
        path = root / f"{name}.npy"
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        assert array.shape == (count, 256, 256, 3) and array.dtype == np.uint8
        actual = digest(path)
        assert actual == manifest[name + "_sha256"], f"Prepared {name} bytes differ"
        checks.append({"split_file": name + ".npy", "sha256": actual, "shape": list(array.shape)})
    sizes = {}
    for split, rows in manifest["splits"].items():
        source = "train" if split == "train" else "val"

        def original(row):
            path = args.shared / f"{source}2017" / row["file"]
            if not path.exists():
                path = args.data_root / "project_sources/coco2017" / f"{source}2017" / row["file"]
            assert digest(path) == row["original_sha256"], (
                f"Original photograph differs: {row['id']}"
            )
            with Image.open(path) as image:
                return min(image.size)

        with concurrent.futures.ThreadPoolExecutor(4) as pool:
            values = np.asarray(list(pool.map(original, rows)))
        sizes[split] = {
            "n": len(values),
            "source_short_side_min": int(values.min()),
            "source_short_side_median": float(np.median(values)),
            "source_short_side_q25": float(np.quantile(values, 0.25)),
            "source_short_side_below256": int((values < 256).sum()),
        }
        print("Verified original bytes and resolution", split, len(values), flush=True)
    write_json(
        args.output,
        {
            "status": "passed",
            "prepared_arrays": checks,
            "splits": sizes,
            "script_sha256": digest(Path(__file__)),
            "protocol": "Every prepared-array and original-photograph SHA256 matches the published manifest. Dimensions read from actual image headers. Center-crop/resize to256 enlarges originals with shorter sides below256; no identities or pixels were changed by this descriptive verification.",
        },
    )


if __name__ == "__main__":
    main()
