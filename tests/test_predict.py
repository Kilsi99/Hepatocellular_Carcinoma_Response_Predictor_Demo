import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from fastapi.testclient import TestClient
from Backend.main import app       


client = TestClient(app)

fake_patient = {
    # Liver function / enzymes
    "ast": 35.0,
    "alt": 40.0,
    "alp": 90.0,
    "albumin": 4.2,
    "total_bilirubin": 1.0,
    "afp": 15.0,

    # Tumor / disease burden
    "stage_at_diagnosis": 2,
    "t_stage_at_diagnosis": 3,

    # Demographics
    "age": 58,
    "gender": 1,  # Male

    # Comorbidities / history
    "pmh_cirrhosis": 1,
    "pmh_fatty_liver": 0,
    "comorbid_diabetes": 0,
    "comorbid_htn": 1,
    "comorbid_cad": 0,

    # Systemic therapy regimens
    "regimen_atezo_bev": 1,
    "regimen_durva_treme": 0,
    "regimen_nivo_ipi": 0,
    "regimen_pembro_ipi": 0,

    # Local liver treatments
    "local_treatment_given_TACE": 1,
    "local_treatment_given_Y90": 0,
    "local_treatment_given_RFA": 0,
    "local_treatment_given_None": 0,

    # Other treatments
    "neoadjuvant_therapy": 0,
    "adjuvant_treatment_given": 0,

    # Free text (optional)
    "clinical_notes": "Patient has mild liver enzyme elevation, no significant comorbidities."
}

edge_patient = {
    "ast": 0.0,
    "alt": 999.0,
    "alp": 0.0,
    "albumin": 0.0,
    "total_bilirubin": 0.0,
    "afp": 0.0,
    "stage_at_diagnosis": 0,
    "t_stage_at_diagnosis": 0,
    "age": 120,
    "gender": 0,
    "pmh_cirrhosis": 0,
    "pmh_fatty_liver": 1,
    "comorbid_diabetes": 1,
    "comorbid_htn": 1,
    "comorbid_cad": 1,
    "regimen_atezo_bev": 0,
    "regimen_durva_treme": 1,
    "regimen_nivo_ipi": 1,
    "regimen_pembro_ipi": 0,
    "local_treatment_given_TACE": 0,
    "local_treatment_given_Y90": 1,
    "local_treatment_given_RFA": 1,
    "local_treatment_given_None": 0,
    "neoadjuvant_therapy": 1,
    "adjuvant_treatment_given": 1,
    "clinical_notes": "Extreme edge case patient"
}

failure_patient = {
    # Liver function / enzymes: wrong types (strings instead of floats)
    "ast": "thirty-five",  # should be float
    "alt": "forty",        # should be float
    "alp": None,           # missing value
    "albumin": -1,         # unrealistic negative value (optional check)
    "total_bilirubin": "high", # wrong type
    "afp": "fifteen",      # wrong type

    # Tumor / disease burden
    "stage_at_diagnosis": "second",  # wrong type
    "t_stage_at_diagnosis": None,    # missing value

    # Demographics
    "age": "fifty-eight",  # wrong type
    "gender": 3,            # invalid if only 0/1 allowed

    # Comorbidities / history
    "pmh_cirrhosis": "yes", # wrong type
    "pmh_fatty_liver": "no", # wrong type
    "comorbid_diabetes": -1, # invalid value
    "comorbid_htn": "sometimes", # wrong type
    "comorbid_cad": None,         # missing value

    # Systemic therapy regimens
    "regimen_atezo_bev": "1",    # string instead of int
    "regimen_durva_treme": None, # missing value
    "regimen_nivo_ipi": 2,       # invalid if only 0/1 allowed
    "regimen_pembro_ipi": -1,    # invalid value

    # Local liver treatments
    "local_treatment_given_TACE": "TACE",  # wrong type
    "local_treatment_given_Y90": None,     # missing value
    "local_treatment_given_RFA": "0",      # string instead of int
    "local_treatment_given_None": 1,       # technically ok

    # Other treatments
    "neoadjuvant_therapy": "no",           # wrong type
    "adjuvant_treatment_given": None,      # missing

    # Free text (optional)
    "clinical_notes": 1234                 # wrong type
}

def test_predict_endpoint():

    response = client.post('/api/v1/predict', json= fake_patient)
    json_data = response.json()

    assert response.status_code == 200 
    assert "probability" in json_data
    assert 'prediction' in json_data
    assert 'shap_values' in json_data
    assert 'baseline' in json_data 
    assert 'toxicity_proba' in json_data
    assert 'data' in json_data
    


def test_predict_edge_patient():
    response = client.post("/api/v1/predict", json=edge_patient)
    assert response.status_code == 200
    json_data = response.json()
    assert "probability" in json_data
    assert "prediction" in json_data
    assert "shap_values" in json_data
    assert "baseline" in json_data
    assert "toxicity_proba" in json_data
    assert "data" in json_data
    print("Edge patient response:", json_data)

def test_predict_failure_patient():
    response = client.post("/api/v1/predict", json=failure_patient)

    # FastAPI + Pydantic should reject this input
    assert response.status_code == 422  # Unprocessable Entity
    print("Failure patient response:", response.json())