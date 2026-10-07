"""Keep installed-command defaults identical to declared repository protocols."""

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    target = ROOT / "src/learning_lab/resources/configs"
    target.mkdir(parents=True, exist_ok=True)
    sources = sorted((ROOT / "configs").glob("*.json"))
    if args.check:
        assert {p.name for p in target.glob("*.json")} == {p.name for p in sources}
    for path in sources:
        destination = target / path.name
        if args.check:
            assert path.read_bytes() == destination.read_bytes(), path.name
        else:
            destination.write_bytes(path.read_bytes())
    print("Packaged defaults", "verified" if args.check else "synchronized", len(sources))


if __name__ == "__main__":
    main()
