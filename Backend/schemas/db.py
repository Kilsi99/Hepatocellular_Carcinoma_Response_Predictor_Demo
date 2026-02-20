from pydantic import BaseModel


# =====================================================
# Labs
# =====================================================
class Labs(BaseModel):
    ast: float
    alt: float
    alp: float
    albumin: float
    total_bilirubin: float
    afp: float

# =====================================================
# Patient Data (flat schema, backend-ready)
# =====================================================
class DB_Data(BaseModel):

    # Patient id
    patient_id: int
    # Liver function / enzymes
    ast: float
    alt: float
    alp: float
    albumin: float
    total_bilirubin: float
    afp: float

    # Tumor / disease burden
    stage_at_diagnosis: int
    t_stage_at_diagnosis: int

    # Demographics
    age: int
    gender: int  # 1=Male, 0=Female

    # Comorbidities / history
    pmh_cirrhosis: int
    pmh_fatty_liver: int
    comorbid_diabetes: int
    comorbid_htn: int
    comorbid_cad: int

    # Systemic therapy regimens
    regimen_atezo_bev: int
    regimen_durva_treme: int
    regimen_nivo_ipi: int
    regimen_pembro_ipi: int

    # Local liver treatments
    local_treatment_given_TACE: int
    local_treatment_given_Y90: int
    local_treatment_given_RFA: int
    local_treatment_given_None: int

    # Other treatments
    neoadjuvant_therapy: int
    adjuvant_treatment_given: int

    # NER / derived flags
    liver_tumor_flag: int
    liver_disease_flag: int
    portal_hypertension_flag: int
    biliary_flag: int
    symptoms_flag: int

    # model results 
    prediction: int 
    probability: float 