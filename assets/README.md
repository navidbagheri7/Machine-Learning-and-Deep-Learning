# Generated scientific figures

Figures are generated from completed reference records by `scripts/build_report.py`.
Every figure is supplied as PNG for GitHub and PDF for export. `source_data/` contains
compact numerical arrays used by those figures, including a small deterministic CIFAR
error sample. It contains no executable Python object arrays.

The overview combines measured studies with distinct datasets and protocols; it is
not a common leaderboard. Signal plots use a labeled 1e-6 display floor, while exact
values remain in the metrics. Double-descent bands show interquartile variability,
and bar errors show seed standard deviation where multiple runs exist.
