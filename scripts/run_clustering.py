"""Run the full declared clustering comparison (no class-label selection)."""

import json
from pathlib import Path

from learning_lab.clustering import experiment

if __name__ == "__main__":
    config = json.loads(Path("configs/unsupervised.json").read_text())
    if Path(config["output"], "run.json").exists():
        raise FileExistsError(
            "Preserve existing clustering evidence; choose a fresh protocol/output"
        )
    experiment(config)
