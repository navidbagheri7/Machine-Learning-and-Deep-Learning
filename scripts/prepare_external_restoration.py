"""Fetch official external references into a personal cache; record actual content hashes."""

import argparse
import concurrent.futures
import hashlib
import json
import shutil
import subprocess
import urllib.request
from pathlib import Path

SOURCES = {
    "CT2": ("shuchenweng/CT2", "cd40f5a0cee23365b831f25ab5d659674936018b"),
    "BigColor": ("KIMGEONUNG/BigColor", "5b7687fbedd9af22fef02a82102c6d7708ab1217"),
    "SwinIR": ("JingyunLiang/SwinIR", "6545850fbf8df298df73d81f3e8cba638787c8bd"),
    "BasicSR": ("XPixelGroup/BasicSR", "8d56e3a045f9fb3e1d8872f92ee4a4f07f886b0a"),
}
DRIVE = {
    "CT2/release.rar": "15LsqvHu1_g6OEEUbir7i1wbLDxIP4LTU",
    "BigColor/ckpts/bigcolor/EG_011.ckpt": "1SCrcIxhm2F1SMTUUcNTEtB4fQu6y7RDK",
    "BigColor/ckpts/bigcolor/EG_EMA_011.ckpt": "1A6JwWeedJa56_6lDOkkR3hqvJ_aNUghE",
    "BigColor/ckpts/bigcolor/args.pkl": "1iQ2OAnQRAqd-6PDtQCrqh5-L_F_GXpq4",
    "BigColor/pretrained/config.pickle": "1qvxR2ZK5-gaLgtxem9fdLmfSKKvXF6eW",
}


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url, path):
    """Commit a downloaded file only after the network transfer completes."""
    temporary = path.with_name(path.name + ".part")
    urllib.request.urlretrieve(url, temporary)
    temporary.replace(path)


def prepare(root):
    import gdown

    models = root / "models"
    results = {"sources": {}, "weights": {}, "unavailable": []}
    mislabeled = models / "CT2" / "checkpoint.pth"
    if mislabeled.exists():
        with mislabeled.open("rb") as handle:
            if handle.read(4) == b"Rar!":
                mislabeled.rename(models / "CT2" / "release.rar")
    for name, (repo, rev) in SOURCES.items():
        path = models / name
        if not path.exists():
            subprocess.run(
                ["git", "clone", "https://github.com/" + repo + ".git", str(path)], check=True
            )
        actual = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
        ).strip()
        if actual != rev:
            subprocess.run(["git", "-C", str(path), "fetch", "origin", rev], check=True)
            subprocess.run(["git", "-C", str(path), "checkout", rev], check=True)
        results["sources"][name] = {
            "repository": "https://github.com/" + repo,
            "revision": rev,
            "license": "CC BY-NC 4.0"
            if name == "CT2"
            else ("MIT" if name == "BigColor" else "Apache-2.0"),
        }

    def download(item):
        relative, identifier = item
        path = models / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if not path.exists():
                temporary = path.with_suffix(path.suffix + ".part")
                returned = gdown.download(id=identifier, output=str(temporary), quiet=True)
                if not returned:
                    raise OSError("Official Google Drive file unavailable")
                temporary.replace(path)
            return relative, {
                "url": "https://drive.google.com/file/d/" + identifier,
                "sha256": digest(path),
                "bytes": path.stat().st_size,
            }
        except Exception as error:
            return relative, {"error": str(error)}

    with concurrent.futures.ThreadPoolExecutor(3) as executor:
        for name, record in executor.map(download, DRIVE.items()):
            print(name, record, flush=True)
            if "error" in record:
                results["unavailable"].append({"file": name, **record})
            else:
                results["weights"][name] = record
    archive = models / "CT2" / "release.rar"
    if archive.exists() and not (models / "CT2" / "checkpoint.pth").exists():
        import libarchive

        with libarchive.file_reader(str(archive)) as entries:
            for entry in entries:
                name = Path(entry.pathname).name
                print("CT2 archive entry", entry.pathname, entry.size, flush=True)
                if name.endswith(".pth") or name == "variant.yml":
                    destination = (
                        models / "CT2" / ("checkpoint.pth" if name.endswith(".pth") else name)
                    )
                    temporary = destination.with_name(destination.name + ".part")
                    with temporary.open("wb") as stream:
                        for block in entry.get_blocks():
                            stream.write(block)
                    temporary.replace(destination)
                    results["weights"]["CT2/" + destination.name] = {
                        "url": results["weights"]["CT2/release.rar"]["url"],
                        "archive_member": entry.pathname,
                        "sha256": digest(destination),
                        "bytes": destination.stat().st_size,
                    }
    elif (models / "CT2" / "checkpoint.pth").exists():
        path = models / "CT2" / "checkpoint.pth"
        results["weights"]["CT2/checkpoint.pth"] = {
            "url": results["weights"]["CT2/release.rar"]["url"],
            "sha256": digest(path),
            "bytes": path.stat().st_size,
        }
    releases = {
        "SwinIR/" + filename: "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/"
        + filename
        for filename in [
            "001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth",
            "005_colorDN_DFWB_s128w8_SwinIR-M_noise25.pth",
        ]
    }
    releases["BasicSR/weights/RealESRGAN_x4plus.pth"] = (
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
    )
    for relative, url in releases.items():
        path = models / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            fetch(url, path)
        results["weights"][relative] = {
            "url": url,
            "sha256": digest(path),
            "bytes": path.stat().st_size,
        }
    hub = models / "torch" / "hub" / "checkpoints"
    hub.mkdir(parents=True, exist_ok=True)
    for name in [
        "alexnet-owt-7be5be79.pth",
        "resnet18-f37072fd.pth",
        "resnet50-11ad3fa6.pth",
        "convnext_tiny-983f1562.pth",
    ]:
        path = hub / name
        if not path.exists():
            cached = Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / name
            if cached.exists():
                shutil.copyfile(cached, path)
            else:
                fetch("https://download.pytorch.org/models/" + name, path)
        results["weights"]["torch/hub/checkpoints/" + name] = {
            "url": "https://download.pytorch.org/models/" + name,
            "sha256": digest(path),
            "bytes": path.stat().st_size,
        }
    expected = Path(__file__).resolve().parents[1] / "assets/external_restoration_manifest.json"
    if expected.exists():
        pins = json.loads(expected.read_text())
        for name, record in results["weights"].items():
            if name in pins["weights"] and record["sha256"] != pins["weights"][name]["sha256"]:
                raise ValueError(f"Downloaded checkpoint differs from the published hash: {name}")
    if results["unavailable"]:
        raise RuntimeError(
            f"Some official checkpoints remain unavailable: {results['unavailable']}"
        )
    (models / "restoration_manifest.json").write_text(json.dumps(results, indent=2) + "\n")
    print("Manifest:", models / "restoration_manifest.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", type=Path, default=Path("/data/navid.bagheri-shouraki/ML_data")
    )
    prepare(parser.parse_args().data_root)
