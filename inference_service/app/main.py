from __future__ import annotations

import os
from typing import Any, Dict, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict

import mlflow


# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
# Prefer MODEL_PATH for local Docker (joblib pipeline saved from training):
#   export MODEL_PATH="/app/artifacts/readmit_baseline_lr.joblib"
#
# Or use MLflow URIs:
#   export MODEL_URI="runs:/<RUN_ID>/model"
#   export MODEL_URI="models:/readmit_baseline_lr/latest"
#
# If both are set, MODEL_PATH wins (most reliable in Docker).
# -----------------------------------------------------------------------------
MODEL_PATH = os.getenv("MODEL_PATH", "").strip()
MODEL_URI = os.getenv("MODEL_URI", "").strip()

# If using local file-based MLflow store inside container, you can optionally set:
# export MLFLOW_TRACKING_URI="file:/app/mlruns"
# export MLFLOW_REGISTRY_URI="file:/app/mlruns"
# (not required for MODEL_PATH mode)


# -----------------------------------------------------------------------------
# Pydantic Schemas
# -----------------------------------------------------------------------------
class PredictRequest(BaseModel):
    age: int = Field(..., ge=0, le=120)
    sex: str = Field(..., min_length=1)
    chief_complaint: str = Field(..., min_length=1)

    hr: int = Field(..., ge=0, le=250)
    sbp: int = Field(..., ge=0, le=300)
    dbp: int = Field(..., ge=0, le=200)

    wbc: float = Field(..., ge=0, le=50)
    lactate: float = Field(..., ge=0, le=20)

    los_days: int = Field(..., ge=0, le=60)


class PredictResponse(BaseModel):
    # Avoid "model_" namespace warning by using model_ref instead of model_uri
    model_config = ConfigDict(protected_namespaces=())

    model_ref: str
    readmit_probability: float
    readmit_pred: int
    metadata: Dict[str, Any] = {}


# -----------------------------------------------------------------------------
# Model Loading
# -----------------------------------------------------------------------------
def load_model():
    """
    Loads either:
      1) joblib model from MODEL_PATH (recommended for local Docker)
      2) MLflow model from MODEL_URI (runs:/ or models:/)
    """
    if MODEL_PATH:
        import joblib

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"MODEL_PATH not found: {MODEL_PATH}. "
                f"Did you mount your artifacts folder into the container?"
            )
        return joblib.load(MODEL_PATH), f"path:{MODEL_PATH}"

    if MODEL_URI:
        model = mlflow.pyfunc.load_model(MODEL_URI)
        return model, f"uri:{MODEL_URI}"

    raise ValueError("You must set either MODEL_PATH or MODEL_URI.")


def predict_proba(model, X: pd.DataFrame) -> float:
    """
    Returns probability of readmit=1.
    Works for:
      - sklearn pipeline (joblib loaded) -> has predict_proba
      - mlflow.pyfunc model (may not expose predict_proba) -> try multiple strategies
    """
    # Case 1: sklearn pipeline
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(X)[0, 1])

    # Case 2: MLflow pyfunc: try predict(), interpret output
    yhat = model.predict(X)

    # yhat could be numpy array, list, pandas Series/DataFrame
    try:
        # If it’s a vector of probabilities
        return float(yhat.iloc[0]) if hasattr(yhat, "iloc") else float(yhat[0])
    except Exception:
        # If it’s a class prediction
        pred_class = int(yhat.iloc[0]) if hasattr(yhat, "iloc") else int(yhat[0])
        return float(pred_class)


# -----------------------------------------------------------------------------
# FastAPI App
# -----------------------------------------------------------------------------
app = FastAPI(title="MediScaleAI Inference", version="0.2.0")

MODEL = None
MODEL_REF = ""


@app.on_event("startup")
def startup():
    global MODEL, MODEL_REF
    MODEL, MODEL_REF = load_model()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "loaded": MODEL is not None,
        "model_ref": MODEL_REF or None,
        "model_path": MODEL_PATH or None,
        "model_uri": MODEL_URI or None,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    X = pd.DataFrame(
        [{
            "sex": req.sex,
            "chief_complaint": req.chief_complaint,
            "age": req.age,
            "hr": req.hr,
            "sbp": req.sbp,
            "dbp": req.dbp,
            "wbc": req.wbc,
            "lactate": req.lactate,
            "los_days": req.los_days,
        }]
    )

    prob = predict_proba(MODEL, X)
    pred = 1 if prob >= 0.5 else 0

    return PredictResponse(
        model_ref=MODEL_REF,
        readmit_probability=float(prob),
        readmit_pred=int(pred),
        metadata={"threshold": 0.5},
    )