"""Dataset loaders. Downloads are explicit; training never downloads implicitly."""

import hashlib
import io
import json
import urllib.request
from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_california_housing

ADULT_COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income",
]
NUMERIC = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
CATEGORICAL = [
    "workclass",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
]


def prepare(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name in ["adult.data", "adult.test"]:
        url = "https://archive.ics.uci.edu/static/public/2/adult.zip"
        # The archive contains the original fixed training and test splits.
        if not (root / name).exists():
            import zipfile

            with urllib.request.urlopen(url, timeout=120) as response:
                archive = response.read()
            with zipfile.ZipFile(io.BytesIO(archive)) as z:
                (root / name).write_bytes(z.read(name))
        manifest[name] = {
            "source": url,
            "sha256": hashlib.sha256((root / name).read_bytes()).hexdigest(),
        }
    data = fetch_california_housing(data_home=root, download_if_missing=True)
    manifest["california_housing"] = {
        "source": "https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html",
        "rows": len(data.target),
        "features": data.feature_names,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def adult(root):
    def read(name):
        df = pd.read_csv(
            Path(root) / name,
            names=ADULT_COLUMNS,
            skipinitialspace=True,
            comment="|",
            na_values="?",
        )
        y = df.pop("income").str.strip().str.rstrip(".").eq(">50K").astype(int).to_numpy()
        # education_num duplicates education; fnlwgt is a survey weight, not a predictor.
        return df[NUMERIC + CATEGORICAL], y

    return read("adult.data"), read("adult.test")


def housing(root):
    data = fetch_california_housing(data_home=root, download_if_missing=False, as_frame=True)
    return data.data, data.target.to_numpy()
