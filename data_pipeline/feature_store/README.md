# Feature Store (v1)

This folder contains encounter-level features produced by Spark ETL.

Output location (default):
- `data/processed/features_encounter/` (Parquet)

Schema includes:
- vitals aggregates (mean/std/min/max)
- last observed vitals
- note features + raw note text for BERT fine-tuning
- labels for supervised training