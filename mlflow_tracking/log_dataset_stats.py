from __future__ import annotations

import mlflow
import pandas as pd

FEATURE_SAMPLE = "data/raw/ehr_visits.csv"

def main():
    df = pd.read_csv(FEATURE_SAMPLE)
    
    mlflow.set_experiment("mediscale-data")
    with mlflow.start_run(run_name="raw_dataset_stats"):
        mlflow.log_param("rows", len(df))
        mlflow.log_param("cols", df.shape[1])
        
        # basic label rates
        mlflow.log_metric("readmit_30d_rate", float(df["readmit_30d"].mean()))
        mlflow.log_mertic("mortality_rate", float[df["mortality_inpatient"].mean()])
        mlflow.log_metric("avg_los", float(df["los_days"].mean()))
        
    print("✅ Logged stats to MLflow (experiment: mediscale-data)")
    
if __name__ == "main":
    main()