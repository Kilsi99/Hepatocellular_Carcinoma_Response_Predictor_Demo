from fastapi import APIRouter, HTTPException
import numpy as np
import pathlib
import os


from Backend.schemas.patient import PatientData
from Backend.schemas.Results import ResultsData
from Backend.schemas.db import DB_Data
from Backend.schemas.Patientcheck import Patient_check
from Backend.schemas.Outcome import OutcomeCreate

from Backend.services.model_service import HCCModelService, ToxicityModelService
from Backend.services.Feature_service import build_features
from Backend.services.explanation_service import ExplanationService

import traceback


# -----------------------------
# FastAPI router
# -----------------------------
router = APIRouter()

# -----------------------------
# Paths to models
# -----------------------------
base_path = pathlib.Path("C:/Users/kilsi/OneDrive/Documents/Curenetics/HCC_APP_Demo/models")
hcc_model_path = base_path / "random_forest_demo.pkl"
toxicity_model_path = base_path / "logistic_toxicity_model.pkl"
toxicity_scaler_path = base_path / "logistic_toxicity_scaler.pkl"

# -----------------------------
# Initialize services separately
# -----------------------------
hcc_service = HCCModelService(str(hcc_model_path))
toxicity_service = ToxicityModelService(str(toxicity_model_path), str(toxicity_scaler_path))
explanation_service = ExplanationService()

@router.post("/predict")
def predict(patient: PatientData):
    try:
        # Build features
        X = build_features(patient).reshape(1, -1)

        # HCC prediction
        proba = hcc_service.hcc_predict(X)
        prediction = int(proba[0, 1] > 0.5)

        # SHAP explanation
        shap_values, base_value = hcc_service.explain_prediction(X)

        # Toxicity prediction
        toxicity_proba = toxicity_service.predict_toxicity(X)
        toxicity_probs_flat = toxicity_proba[0].tolist()
        print(toxicity_proba)

        return {
            "probability": float(proba[0, 1]),
            "prediction": prediction,
            "shap_values": shap_values.tolist(),
            "baseline": base_value,
            "toxicity_proba": toxicity_probs_flat,  # handles multi-class arrays
            "data": X.tolist()
        }

    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@router.post('/explanation')
def explain(results: ResultsData):
    shap_values_list = results.shap_values
    probability = results.probability

    explanation = explanation_service.generate_explanation(probability, shap_values_list)
    return {'explanation': explanation}


@router.get("/health")
def health_check():
    return {
        "status": "ok",
        "hcc_model_loaded": True,
        "toxicity_model_loaded": True,
        "database_connected": True,
        "db_error": None
    }


@router.post("/insert_data")
def insert_data(db_info: DB_Data):
    return {
        "success": True,
        "data": db_info.dict(),  # just echo back what was sent
        "message": "Dummy save successful"
    }


@router.post("/check_data")
def check_data(data: Patient_check):
    return {
        "exists": True,
        "patient_id": data.patient_id
    }


@router.post("/insert_outcome")
def insert_outcome(data: OutcomeCreate):
    return {
        "status": "success",
        "inserted": {
            "patient_id": data.patient_id,
            "outcome": 1 if data.outcome else 0
        }
    }