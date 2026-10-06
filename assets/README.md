# Generated scientific figures

Figures are generated from completed reference records by `scripts/build_report.py`.
Every figure is supplied as PNG for GitHub and PDF for export. `source_data/` contains
compact numerical arrays used by those figures, including a small deterministic CIFAR
error sample. It contains no executable Python object arrays.

The overview combines measured studies with distinct datasets and protocols; it is
not a common leaderboard. Signal plots use a labeled 1e-6 display floor, while exact
values remain in the metrics. Double-descent bands show interquartile variability,
and bar errors show seed standard deviation where multiple runs exist.

Project extensions add mixture/inverse diagnostics, training curves and restoration
galleries. Natural-photo figure inputs retain CC BY 2.0; [photo attributions](../docs/photo_attributions.md)
and `photo_credits.json` identify creators, source URLs and modifications.
Only the first four test-order photos with verified live attribution are published;
all 400 declared test images contribute to evaluation. MNIST digits and supplied
synthetic coordinate values are separately identified. No handout or raw archive is included.

Losslessly compressed `*_per_image_scores.npz` files contain PSNR vectors keyed by
the corresponding JSON rows, supporting paired comparisons without expanding the release
with repeated numerical lists. `project_data_manifest.json` records source and cache hashes,
archive dimensions/missing IDs, and exact natural-image identities.
`pretrained_manifest.json` pins the frozen DDColor source/model revisions, licence and weight hash.
