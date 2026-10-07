# Observation schemas

These two CSV files contain fictional observations for exercising saved-model commands.
They do not contain coursework, raw benchmark records or evidence of predictive quality.

After fitting the classical protocol, run:

```bash
python scripts/predict_classical.py --checkpoint results/reference/classical/adult/0/checkpoints/catboost.joblib --metadata assets/classical_calibration/adult_catboost_0.json --input examples/adult_observations.csv --output .local/adult_predictions.csv
python scripts/predict_classical.py --checkpoint results/reference/classical/housing_random/0/checkpoints/catboost.joblib --metadata assets/classical_calibration/housing_random_catboost_0.json --input examples/housing_observations.csv --output .local/housing_predictions.csv
```

Classification returns calibrated probabilities, the ordered decision label, set membership
and explicit empty sets. Regression returns point predictions and split-conformal intervals
in $100,000 units. The original calibration's exchangeability assumptions still apply;
matching a CSV schema alone does not establish reliable coverage for a new population.

The API checks the checkpoint hash before loading it, validates required feature columns,
preserves class order, handles declared missing values and rejects infinities. Unused CSV
columns are ignored. Numeric datasets use their scikit-learn feature names in metadata.
Selected model files stay private; reproduce them with the training commands in
[classical results](../docs/classical_results.md). Export calibration and verify all held-out
predictions with `python scripts/export_classical_inference.py`.
