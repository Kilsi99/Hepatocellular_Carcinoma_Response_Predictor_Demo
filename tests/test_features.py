import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Backend.services.NLP_service import generate_ner_flags
from Backend.services.Feature_service import build_features
from fastapi.testclient import TestClient
from Backend.main import app 
from Backend.schemas.patient import PatientData
import pytest
import numpy as np

client = TestClient(app)


# -------------------------------
# Sample clinical notes for testing
# -------------------------------
sample_note = """
Patient with a history of liver cirrhosis and ascites. Shows signs of hepatocellular carcinoma.
Also has portal hypertension and mild splenomegaly.
"""

empty_note = "Patient with no liver or biliary issues. No tumors detected."

# -------------------------------
# Test: NER flag output shape
# -------------------------------
def test_ner_flag_shape():
    flags = generate_ner_flags(sample_note)
    # Ensure flags is a list of length 5
    assert isinstance(flags, list), "Output should be a list"
    assert len(flags) == 5, f"NER flags should have length 5, got {len(flags)}"


# -------------------------------
# Mock NER to keep test deterministic
# -------------------------------
def mock_generate_ner_flags(clinical_notes):
    # Return fixed flags for testing: liver_tumor, liver_disease, portal_hypertension, biliary, symptoms
    return [1, 1, 0, 0, 1]

def test_long_clinical_note_handling():
    long_note = "Patient has liver carcinoma. " * 200  # >512 tokens
    flags = generate_ner_flags(long_note)
    assert len(flags) == 5
    # Ensure no errors in feature building
    patient = PatientData(
        ast=30, alt=35, alp=80, albumin=4.0, total_bilirubin=1.0, afp=10,
        stage_at_diagnosis=2, t_stage_at_diagnosis=2, age=60, gender=1,
        pmh_cirrhosis=1, pmh_fatty_liver=0, comorbid_diabetes=0,
        comorbid_htn=0, comorbid_cad=0,
        regimen_atezo_bev=1, regimen_durva_treme=0, regimen_nivo_ipi=0,
        regimen_pembro_ipi=0,
        local_treatment_given_TACE=0, local_treatment_given_Y90=0,
        local_treatment_given_RFA=0, local_treatment_given_None=1,
        neoadjuvant_therapy=0, adjuvant_treatment_given=0,
        clinical_notes=long_note
    )
    features = build_features(patient)
    assert features.shape[0] == 30

@pytest.fixture
def sample_patient(monkeypatch):
    # Patch NER function to avoid calling actual model
    monkeypatch.setattr("Backend.services.Feature_service.generate_ner_flags", mock_generate_ner_flags)
    
    return PatientData(
        ast=35.0,
        alt=40.0,
        alp=90.0,
        albumin=4.2,
        total_bilirubin=1.0,
        afp=15.0,
        stage_at_diagnosis=2,
        t_stage_at_diagnosis=3,
        age=58,
        gender=1,
        pmh_cirrhosis=1,
        pmh_fatty_liver=0,
        comorbid_diabetes=0,
        comorbid_htn=1,
        comorbid_cad=0,
        regimen_atezo_bev=1,
        regimen_durva_treme=0,
        regimen_nivo_ipi=0,
        regimen_pembro_ipi=0,
        local_treatment_given_TACE=1,
        local_treatment_given_Y90=0,
        local_treatment_given_RFA=0,
        local_treatment_given_None=0,
        neoadjuvant_therapy=0,
        adjuvant_treatment_given=0,
        clinical_notes="Patient has liver carcinoma and ascites"
    )


# -------------------------------
# Sample clinical notes for testing
# -------------------------------
short_note = """
Patient with a history of liver cirrhosis and ascites. Shows signs of hepatocellular carcinoma.
Also has portal hypertension and mild splenomegaly.
"""

long_note = "Patient has " + "liver carcinoma. " * 200  # very long note

empty_note = "Patient with no liver or biliary issues. No tumors detected."

# -------------------------------
# Helper to check valid NER output
# -------------------------------
def assert_valid_ner_flags(flags):
    assert isinstance(flags, list), "Output should be a list"
    assert len(flags) == 5, f"NER flags should have length 5, got {len(flags)}"
    for f in flags:
        assert f in [0, 1], f"NER flag values should be 0 or 1, got {f}"

# -------------------------------
# Tests
# -------------------------------
def test_ner_flag_shape_short_and_empty():
    assert_valid_ner_flags(generate_ner_flags(short_note))
    assert_valid_ner_flags(generate_ner_flags(empty_note))

def test_ner_long_note():
    assert_valid_ner_flags(generate_ner_flags(long_note))

# -------------------------------
# Test: feature array
# -------------------------------
@pytest.fixture
def sample_patient(monkeypatch):
    # Patch NER to a deterministic mock for features test
    def mock_generate_ner_flags(clinical_notes):
        return [1, 0, 1, 0, 1]
    monkeypatch.setattr("Backend.services.Feature_service.generate_ner_flags", mock_generate_ner_flags)

    return PatientData(
        ast=35.0,
        alt=40.0,
        alp=90.0,
        albumin=4.2,
        total_bilirubin=1.0,
        afp=15.0,
        stage_at_diagnosis=2,
        t_stage_at_diagnosis=3,
        age=58,
        gender=1,
        pmh_cirrhosis=1,
        pmh_fatty_liver=0,
        comorbid_diabetes=0,
        comorbid_htn=1,
        comorbid_cad=0,
        regimen_atezo_bev=1,
        regimen_durva_treme=0,
        regimen_nivo_ipi=0,
        regimen_pembro_ipi=0,
        local_treatment_given_TACE=1,
        local_treatment_given_Y90=0,
        local_treatment_given_RFA=0,
        local_treatment_given_None=0,
        neoadjuvant_therapy=0,
        adjuvant_treatment_given=0,
        clinical_notes="Patient has liver carcinoma and ascites"
    )

def test_build_features_array(sample_patient):
    features = build_features(sample_patient)
    assert isinstance(features, np.ndarray)
    assert features.shape[0] == 30
    # Check NER flags are 0/1
    for f in features[16:21]:
        assert f in [0, 1]


        