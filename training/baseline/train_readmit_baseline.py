from __future__ import annotations

import os
import json
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from sklearn.linear_model import LogisticRegression

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "")
if TRACKING_URI:
    mlflow.set_tracking_uri(TRACKING_URI)

RAW_CSV = "data/raw/ehr_visits.csv"
EXPERIMENT = "mediscale-baseline"
MODEL_NAME = "readmit_baseline_lr"


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # coerce types defensively
    for c in ["age", "hr", "sbp", "dbp", "wbc", "lactate", "los_days", "readmit_30d", "mortality_inpatient"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # simple cleanup
    df = df.dropna(subset=["readmit_30d", "age", "hr", "sbp", "dbp", "wbc", "lactate"])
    df["sex"] = df["sex"].fillna("U")
    df["chief_complaint"] = df["chief_complaint"].fillna("unknown")

    return df


def build_pipeline(cat_cols, num_cols) -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )

    clf = LogisticRegression(max_iter=500, n_jobs=1)
    return Pipeline(steps=[("preprocess", pre), ("model", clf)])


def main():
    assert os.path.exists(RAW_CSV), f"Missing {RAW_CSV}. Run generator first."

    df = load_data(RAW_CSV)

    target = "readmit_30d"
    cat_cols = ["sex", "chief_complaint"]
    num_cols = ["age", "hr", "sbp", "dbp", "wbc", "lactate", "los_days"]

    X = df[cat_cols + num_cols]
    y = df[target].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    mlflow.set_experiment(EXPERIMENT)

    with mlflow.start_run(run_name="baseline_logreg"):
        mlflow.log_param("model", "logistic_regression")
        mlflow.log_param("target", target)
        mlflow.log_param("cat_cols", ",".join(cat_cols))
        mlflow.log_param("num_cols", ",".join(num_cols))
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("test_rows", len(X_test))

        pipe = build_pipeline(cat_cols, num_cols)
        pipe.fit(X_train, y_train)

        proba = pipe.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)

        roc = roc_auc_score(y_test, proba)
        ap = average_precision_score(y_test, proba)
        acc = accuracy_score(y_test, pred)

        mlflow.log_metric("roc_auc", float(roc))
        mlflow.log_metric("avg_precision", float(ap))
        mlflow.log_metric("accuracy", float(acc))

        # Save local artifact + log
        os.makedirs("artifacts", exist_ok=True)
        local_path = "artifacts/readmit_baseline_lr.joblib"
        joblib.dump(pipe, local_path)
        mlflow.log_artifact(local_path)

        # Log model in MLflow
        mlflow.sklearn.log_model(
            sk_model=pipe,
            artifact_path="model",
            registered_model_name=MODEL_NAME
        )

        # Save quick model card
        model_card = {
            "problem": "30-day readmission prediction (synthetic)",
            "model": "LogisticRegression",
            "metrics": {"roc_auc": float(roc), "avg_precision": float(ap), "accuracy": float(acc)},
            "features": {"categorical": cat_cols, "numeric": num_cols},
            "notes": "Baseline model for pipeline validation; replace with distributed training later (Ray/Spark)."
        }
        card_path = "artifacts/model_card.json"
        with open(card_path, "w") as f:
            json.dump(model_card, f, indent=2)
        mlflow.log_artifact(card_path)

        print(f"✅ Logged run to MLflow: {EXPERIMENT}")
        print(f"✅ Metrics: roc_auc={roc:.3f} ap={ap:.3f} acc={acc:.3f}")
        print(f"✅ Registered model name: {MODEL_NAME}")


if __name__ == "__main__":
    main()