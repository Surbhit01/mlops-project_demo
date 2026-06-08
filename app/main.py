from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from app.schemas import EmployeeInput, PredictionOutput
from app.model_loader import load_from_pkl
import pandas as pd


app = FastAPI(
    title="AttritionGuard API",
    description="Predict employee attrition risk",
    version="1.0.0"
)

# ── Load both models at startup ────────────────────────────────────
# mlflow_model = load_from_mlflow("AttritionGuard", version="1")
local_model  = load_from_pkl("models/model.pkl")


# ── Health check ───────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "healthy"}


# ── Predict using MLflow model ─────────────────────────────────────
@app.post("/predict/mlflow", response_model=PredictionOutput)
def predict_mlflow(employee: EmployeeInput):
    """Predict using model loaded from MLflow registry."""
    try:
        input_df     = pd.DataFrame([employee.model_dump()])
        probability  = float(mlflow_model.predict_proba(input_df)[0][1])
        return PredictionOutput(
            attrition_risk = "High" if probability >= 0.5 else "Low",
            probability    = round(probability, 4),
            model_source   = "MLflow Registry — version 1"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Predict using local pkl model ─────────────────────────────────
@app.post("/predict/local", response_model=PredictionOutput)
def predict_local(employee: EmployeeInput):
    """Predict using model loaded from local pickle file."""
    try:
        input_df     = pd.DataFrame([employee.model_dump()])
        probability  = float(local_model.predict_proba(input_df)[0][1])
        return PredictionOutput(
            attrition_risk = "High" if probability >= 0.5 else "Low",
            probability    = round(probability, 4),
            model_source   = "Local file — models/model.pkl"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── SageMaker required endpoints ──────────────────────────────────

@app.get("/ping")
def ping():
    """SageMaker health check — must return 200."""
    return Response(status_code=200)


@app.post("/invocations")
def invocations(employee: EmployeeInput):
    """SageMaker inference endpoint."""
    try:
        input_df       = pd.DataFrame([employee.model_dump()])
        probability    = float(local_model.predict_proba(input_df)[0][1])
        attrition_risk = "High" if probability >= 0.5 else "Low"
        return {
            "attrition_risk": attrition_risk,
            "probability"   : round(probability, 4),
            "model_source"  : "SageMaker endpoint"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))