from fastapi import APIRouter, HTTPException
from Backend.services.model_service import ModelService
from Backend.services.Feature_service import build_features
from Backend.schemas.patient import PatientData


# Create a router
router = APIRouter()

# Instantiate your model service (load model once)
model_service = ModelService(run_id="03949aaab6124cb0800599c1209e45c8")

@router.post('/predict')
def predict(patient: PatientData):
    try:
        # Build the full feature vector from the patient object
        X = build_features(patient)  # pass the whole PatientData object

        # Run model inference
        pred_proba = model_service.predict(X)

        # Return probability and binary prediction
        return {
            "probability": float(pred_proba[0, 1]),
            "prediction": int(pred_proba[0, 1] > 0.5)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))