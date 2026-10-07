"""Run a declared classical benchmark case without modifying the frozen studies."""

import argparse
import json

from threadpoolctl import threadpool_limits

from learning_lab.classical import fit_classification, fit_regression


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/classical.json")
    parser.add_argument(
        "--dataset",
        choices=["adult", "breast_cancer", "digits", "housing_random", "housing_geographic"],
        required=True,
    )
    parser.add_argument("--seed", type=int, choices=[0, 1, 2], required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    config = json.load(open(args.config))
    if args.data_root:
        config["data_root"] = args.data_root
    if args.quick:
        config |= dict(quick=True, trees=8, stacking_folds=2, output="results/smoke/classical")
    with threadpool_limits(config["threads"]):
        if args.dataset.startswith("housing_"):
            fit_regression(config, args.dataset.removeprefix("housing_"), args.seed)
        else:
            fit_classification(config, args.dataset, args.seed)


if __name__ == "__main__":
    main()
