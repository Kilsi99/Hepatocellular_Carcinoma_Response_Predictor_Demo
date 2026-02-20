from Backend.schemas.patient import PatientData 
from Backend.services.NLP_service import generate_ner_flags
import numpy as np 
import pandas as pd 


def build_features(patient: PatientData) -> np.ndarray:
    """
    Build feature array for HCC model in the exact training order.
    Assumes all fields except clinical_notes are already numeric / encoded.
    """

    # --------------------------
    # NER flags
    # --------------------------
    ner_flags = generate_ner_flags(patient.clinical_notes or "")
    liver_tumor_flag = ner_flags[0]
    liver_disease_flag = ner_flags[1]
    portal_hypertension_flag = ner_flags[2]
    biliary_flag = ner_flags[3]
    symptoms_flag = ner_flags[4]

    # --------------------------
    # Feature array in training order
    # --------------------------
    feature_array = np.array([
        # Labs
        patient.ast,
        patient.alt,
        patient.alp,
        patient.albumin,
        patient.total_bilirubin,
        patient.afp,

        # Tumor / disease burden
        patient.stage_at_diagnosis,
        patient.t_stage_at_diagnosis,

        # Demographics
        patient.age,
        patient.gender,

        # Comorbidities / history
        patient.pmh_cirrhosis,
        patient.pmh_fatty_liver,
        patient.comorbid_diabetes,
        patient.comorbid_htn,
        patient.comorbid_cad,

        # NER flags
        liver_tumor_flag,
        liver_disease_flag,        
        portal_hypertension_flag,
        biliary_flag,
        symptoms_flag,

        # Systemic therapy regimens (already 0/1)
        patient.regimen_atezo_bev,
        patient.regimen_durva_treme,
        patient.regimen_nivo_ipi,
        patient.regimen_pembro_ipi,

        # Local liver treatments (already 0/1)
        patient.local_treatment_given_TACE,
        patient.local_treatment_given_Y90,
        patient.local_treatment_given_RFA,
        patient.local_treatment_given_None,

        # Other treatments (already 0/1)
        patient.neoadjuvant_therapy,
        patient.adjuvant_treatment_given
    ], dtype=float)

    return feature_array



