"""Download tabular reference data to an explicitly selected cache."""

import argparse

from learning_lab.data import prepare

parser = argparse.ArgumentParser()
parser.add_argument("--data-root", default="/data/navid.bagheri-shouraki/ML_data")
args = parser.parse_args()
print(prepare(args.data_root))
