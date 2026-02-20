import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import requests
import matplotlib.pyplot as plt
import numpy as np


# =====================================================
# Session state initialization
# =====================================================
for key, default in {
    "result": None,
    "has_prediction": False,
    "explanation_text": None,
    "explanation_requested": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# =====================================================
# App header (ALWAYS TOP)
# =====================================================
st.title("Hepatocellular Carcinoma Response Predictor")

st.markdown("""
## HCC Prediction & Outcome Tracking 🧬

This application is designed for clinicians and researchers to **predict the likelihood of success for different HCC treatment regimens** and to **track real patient outcomes**.  

**What you can do with this app:**
- Input patient **laboratory results** and **demographics**  
- Provide **tumor characteristics**, **comorbidities**, and **past medical history**  
- Include **free-text clinical notes** for additional context  
- Generate a **treatment success prediction** using a machine learning model  
- View **SHAP feature contributions** to understand which patient features influenced the prediction  
- Predict **toxicity risk** for various organ systems  
- Upload **actual patient outcomes** to the database to track whether predicted treatment plans were successful  

**How it works:**
1. Enter patient information in the sidebar.  
2. Click **Run Prediction** to get:  
   - Probability of treatment success  
   - Binary prediction (success/failure)  
   - SHAP-based feature contributions  
   - Predicted toxicity probabilities per organ system  
3. After treatment, use the **Upload Outcome** section to save the real-world outcome linked to the patient's ID.  

*Inference, explanations, and database operations are handled by backend services via secure API endpoints.*
""")

if "system_status" not in st.session_state:
    try:
        r = requests.get("http://localhost:8000/api/v1/health", timeout=5)
        st.session_state.system_status = r.json()
    except Exception as e:
        st.session_state.system_status = {
            "status": "unhealthy",
            "model_loaded": False,
            "database_connected": False,
            "db_error": str(e)
        }

status = st.session_state.system_status

# Model status
if status.get("hcc_model_loaded") and status.get("toxicity_model_loaded"):
    st.success("🟢 Models loaded and ready for predictions")
else:
    st.error("🔴 Models not loaded — predictions unavailable")

# Database status
if status.get("database_connected"):
    st.success("🟢 Database connected")
else:
    st.error(f"🔴 Database connection failed: {status.get('db_error', 'Unknown error')}")

# Overall system status
if status.get("status") != "ok":
    st.warning("⚠️ System is not fully ready. Some features may be disabled.")
    st.stop()  # Stop rendering the rest

# =====================================================
# Encoding maps
# =====================================================
stage_dict = {'IA': 1, 'IB': 2, 'II': 3, 'IIIA': 4, 'IIIB': 5, 'IVA': 6, 'IVB': 7}
t_stage_dict = {'1': 1, '1a': 1, '1b': 1, '2': 2, '3': 3, '4': 4, 'x': 0}




# =====================================================
# Feature names (SHAP) — MUST MATCH MODEL ORDER
# =====================================================
feature_names = [
    'ast', 'alt', 'alp', 'albumin', 'total_bilirubin', 'afp',
    'stage_at_diagnosis', 't_stage_at_diagnosis', 'age', 'gender',
    'pmh_cirrhosis', 'pmh_fatty_liver', 'comorbid_diabetes', 'comorbid_htn', 'comorbid_cad',
    'liver_tumor_flag', 'liver_disease_flag', 'portal_hypertension_flag', 'biliary_flag', 'symptoms_flag',
    'regimen_atezo_bev', 'regimen_durva_treme', 'regimen_nivo_ipi', 'regimen_pembro_ipi',
    'local_treatment_given_TACE', 'local_treatment_given_Y90', 'local_treatment_given_RFA', 'local_treatment_given_None',
    'neoadjuvant_therapy', 'adjuvant_treatment_given'
]

# =====================================================
# Upload Patient Outcome (Standalone Section)
# =====================================================
st.markdown("---")
st.subheader("Upload Patient Outcome to Database 📝")

st.markdown("""
This section allows clinicians or researchers to **upload treatment outcomes** for patients whose HCC predictions have already been generated.  

**Purpose:**  
- Track whether a predicted treatment plan was successful or not.  
- Link outcomes to the correct patient via a **unique Patient ID**.  
- Keep your database up-to-date for future analysis or model retraining.  

**Requirements:**  
1. **Patient ID:** Must match an existing patient in the database (from previous predictions).  
2. **Outcome:** Select whether the treatment plan was successful (`Yes`) or not (`No`).  

**How it works:**  
- The app will first **check if the Patient ID exists** in the predictions table.  
- If it exists, the outcome is uploaded to the `outcomes` table in your database.  
- If the Patient ID is not found, the upload will not proceed and you will be prompted to check the ID.  
""")

# Patient identifier
patient_id = st.text_input(
    "Patient ID",
    placeholder="Enter unique patient identifier",
    help="Must match a Patient ID already in the predictions table"
)

# Outcome selection
outcome = st.selectbox(
    "Was the treatment plan successful?",
    ["No", "Yes"]
)

if st.button("Check Patient and Save Outcome"):
    if not patient_id:
        st.warning("Please enter a Patient ID")
    else:
        with st.spinner("Checking if patient exists..."):
            try:
                # Call backend endpoint to check if patient exists
                r_check = requests.post(
                    "http://localhost:8000/api/v1/check_data",
                    json={"patient_id": patient_id},
                    timeout=10
                )

                if r_check.status_code == 200 and r_check.json().get("exists"):
                    st.success("Patient found in database — ready to save outcome")

                    # Now save outcome
                    payload = {
                        "patient_id": patient_id,
                        "outcome": 1 if outcome == "Yes" else 0
                    }

                    r_save = requests.post(
                        "http://localhost:8000/api/v1/insert_outcome",
                        json=payload,
                        timeout=10
                    )

                    if r_save.status_code == 200:
                        st.success("Outcome successfully saved")
                    else:
                        st.error(f"Failed to save outcome: {r_save.text}")
                else:
                    st.error("Patient ID not found in predictions database")

            except Exception as e:
                st.error(f"Connection error: {e}")

# =====================================================
# Sidebar inputs
# =====================================================
with st.sidebar:
    st.header("Patient Inputs")
    patient_id = st.number_input('Patient id', placeholder='enter unique patient indentifer')

    # ======================
    # Demographics
    # ======================
    st.subheader("Demographics 👤")
    age = st.number_input("Age", 0, 120, 60)
    gender = st.selectbox("Sex at Birth", ["Male", "Female"])

    st.markdown("---")

    # ======================
    # Laboratory Results
    # ======================
    st.subheader("Laboratory Results 🧪")
    ast = st.number_input("AST (U/L)", 0, 1000, 30)
    alt = st.number_input("ALT (U/L)", 0, 1000, 25)
    alp = st.number_input("ALP (U/L)", 0, 1000, 80)
    albumin = st.number_input("Albumin (g/L)", 0.0, 60.0, 40.0)
    total_bilirubin = st.number_input("Total Bilirubin (mg/dL)", 0.0, 30.0, 1.0)
    afp = st.number_input("AFP (ng/mL)", 0.0, 100000.0, 10.0)

    st.markdown("---")

    # ======================
    # Tumour Characteristics
    # ======================
    st.subheader("Tumour Characteristics 🧬")
    stage = st.selectbox("Stage at Diagnosis", list(stage_dict.keys()))
    t_stage = st.selectbox("T-stage", list(t_stage_dict.keys()))

    st.markdown("---")

    # ======================
    # Systemic Therapy (mutually exclusive)
    # ======================
    st.subheader("Systemic Therapy 💊")
    systemic_regimen = st.selectbox(
        "Systemic Regimen",
        [
            "None",
            "Atezolizumab + Bevacizumab",
            "Durvalumab + Tremelimumab",
            "Nivolumab + Ipilimumab",
            "Pembrolizumab + Ipilimumab"
        ]
    )

    st.markdown("---")

    # ======================
    # Local Liver Treatment
    # ======================
    st.subheader("Local Liver Treatment 🧬")
    local_treatment = st.selectbox(
        "Local Treatment",
        ["None", "TACE", "Y90", "RFA"]
    )

    st.markdown("---")

    # ======================
    # Comorbidities / History
    # ======================
    st.subheader("Past Medical History ⚕️")
    cirrhosis = st.selectbox("Cirrhosis", ["No", "Yes"])
    fatty_liver = st.selectbox("Fatty Liver Disease", ["No", "Yes"])
    diabetes = st.selectbox("Diabetes", ["No", "Yes"])
    htn = st.selectbox("Hypertension", ["No", "Yes"])
    cad = st.selectbox("Coronary Artery Disease", ["No", "Yes"])

    st.markdown("---")

    # ======================
    # Other Treatments
    # ======================
    st.subheader("Treatment Timing")
    neoadjuvant = st.selectbox("Neoadjuvant Therapy Given?", ["No", "Yes"])
    adjuvant = st.selectbox("Adjuvant Therapy Given?", ["No", "Yes"])

    st.markdown("---")

    # ======================
    # Clinical Notes
    # ======================
    st.subheader("Clinical Notes 📝")
    clinical_notes = st.text_area(
        "Enter history, symptoms, imaging reports...",
        placeholder="Type patient notes here..."
    )

    st.markdown("---")

    run_prediction = st.button("Run Prediction")


# =====================================================
# Encode inputs for payload
# =====================================================

# Gender
gender_encoded = 1 if gender == "Male" else 0

# Systemic regimen (mutually exclusive)
regimen_atezo_bev = int(systemic_regimen == "Atezolizumab + Bevacizumab")
regimen_durva_treme = int(systemic_regimen == "Durvalumab + Tremelimumab")
regimen_nivo_ipi = int(systemic_regimen == "Nivolumab + Ipilimumab")
regimen_pembro_ipi = int(systemic_regimen == "Pembrolizumab + Ipilimumab")

# Local liver treatments
local_TACE = int(local_treatment == "TACE")
local_Y90 = int(local_treatment == "Y90")
local_RFA = int(local_treatment == "RFA")
local_None = int(local_treatment == "None")


# =====================================================
# Build payload
# =====================================================
payload = {
    # Liver function / enzymes
    "ast": ast,
    "alt": alt,
    "alp": alp,
    "albumin": albumin,
    "total_bilirubin": total_bilirubin,

    # Tumor / disease burden
    "afp": afp,
    "stage_at_diagnosis": stage_dict[stage],
    "t_stage_at_diagnosis": t_stage_dict[t_stage],

    # Demographics
    "age": int(age),
    "gender": gender_encoded,

    # Comorbidities / history
    "pmh_cirrhosis": int(cirrhosis == "Yes"),
    "pmh_fatty_liver": int(fatty_liver == "Yes"),
    "comorbid_diabetes": int(diabetes == "Yes"),
    "comorbid_htn": int(htn == "Yes"),
    "comorbid_cad": int(cad == "Yes"),

    # Systemic therapy regimens
    "regimen_atezo_bev": regimen_atezo_bev,
    "regimen_durva_treme": regimen_durva_treme,
    "regimen_nivo_ipi": regimen_nivo_ipi,
    "regimen_pembro_ipi": regimen_pembro_ipi,

    # Local liver treatments
    "local_treatment_given_TACE": local_TACE,
    "local_treatment_given_Y90": local_Y90,
    "local_treatment_given_RFA": local_RFA,
    "local_treatment_given_None": local_None,

    # Other treatments
    "neoadjuvant_therapy": int(neoadjuvant == "Yes"),
    "adjuvant_treatment_given": int(adjuvant == "Yes"),

    # Free text (NER pipeline)
    "clinical_notes": clinical_notes
}


# =====================================================
# Run prediction
# =====================================================
if run_prediction:
    with st.spinner("Running model inference..."):
        r = requests.post("http://localhost:8000/api/v1/predict", json=payload)

    if r.status_code == 200:
        st.session_state.result = r.json()
        st.session_state.has_prediction = True
        st.session_state.explanation_text = None
        st.session_state.explanation_requested = False
        st.success("Prediction complete")
    else:
        st.error("Prediction failed")
        st.stop()


# =====================================================
# Prediction output (STAYS VISIBLE)
# =====================================================
if st.session_state.has_prediction:
    result = st.session_state.result

    st.markdown("---")
    st.subheader("Model Prediction")

    # ---------------------------
    # HCC Response Prediction
    # ---------------------------
    if result["prediction"] == 1:
        st.write(
            f"The model predicted that the immunotherapy regimen would be successful "
            f"with a predicted response probability of {result['probability']:.2%}"
        )
    else:
        st.write(
            f"The model predicted that the immunotherapy regimen would NOT be successful "
            f"(predicted response probability: {result['probability']:.2%})"
        )

    st.markdown("Below is a feature contribution plot (SHAP)")

    # ---------------------------
    # SHAP feature contributions
    # ---------------------------
    try:
        shap_values = np.array(result["shap_values"])
        base_value = float(result["baseline"])

        shap_df = pd.DataFrame({
            "Feature": feature_names,
            "SHAP value": shap_values
        }).sort_values(by="SHAP value", key=abs, ascending=False)

        # Display bar chart directly
        st.bar_chart(shap_df.set_index("Feature"))

    except Exception as e:
        st.error(f"SHAP explanation failed: {e}")


    # ---------------------------
    # Toxicity ProbabilitiesS
    # ---------------------------
    st.markdown("---")
    st.subheader("Predicted Toxicity Probabilities ⚠️")

    organ_systems = [
        'skin', 'liver', 'gi', 'lung', 'endocrine',
        'neurologic', 'oral', 'musculoskeletal', 'renal', 'other'
    ]

    toxicity_probs = result.get("toxicity_proba", [0]*len(organ_systems))

    # Define threshold for “higher association” (e.g., >5%)
    threshold = 0.05

    # Keep only organs above threshold and exclude 'other'
    organs_with_risk = [
        (organ.capitalize(), prob)
        for organ, prob in zip(organ_systems, toxicity_probs)
        if prob > threshold and organ != "other"
    ]

    if organs_with_risk:
        st.write(
            "Based on this patient's data, the following organ systems have a higher association "
            "with potential adverse events:"
        )

        for organ, prob in organs_with_risk:
            st.write(f"- **{organ}**: has a higher association with adverse events")
    else:
        st.write("No organ systems show a meaningful predicted risk for adverse events in this patient.")


# =====================================================
# Explanation section (CLEAN + STABLE)
# =====================================================
if st.session_state.has_prediction:
    st.markdown("---")
    st.subheader("Model Explanation")

    if st.button("Provide explanation") and not st.session_state.explanation_requested:
        st.session_state.explanation_requested = True

        with st.spinner("Generating explanation..."):
            r = requests.post(
                "http://localhost:8000/api/v1/explanation",
                json={
                    "probability": round(result["probability"], 2),
                    "predicted_class": result["prediction"],
                    "shap_values": result["shap_values"],
                },
                timeout=180
            )

        if r.status_code == 200:
            st.session_state.explanation_text = r.json()["explanation"]
        else:
            st.error("Explanation failed")

    # Display explanation as plain text
    if st.session_state.explanation_text:
        st.write(st.session_state.explanation_text)

if st.session_state.has_prediction:
    result = st.session_state.result
    data_array = result["data"][0]  # data is wrapped in a list from X.tolist()

    # Extract NER flags from positions 10–14
    liver_tumor_flag = int(data_array[10])
    liver_disease_flag = int(data_array[11])
    portal_hypertension_flag = int(data_array[12])
    biliary_flag = int(data_array[13])
    symptoms_flag = int(data_array[14])

    # Prepare DB payload
    db_payload = {
        
        # Patient id 
        'patient_id': patient_id,
        # Liver function / enzymes
        "ast": ast,
        "alt": alt,
        "alp": alp,
        "albumin": albumin,
        "total_bilirubin": total_bilirubin,
        "afp": afp,

        # Tumor / disease burden
        "stage_at_diagnosis": stage_dict[stage],
        "t_stage_at_diagnosis": t_stage_dict[t_stage],

        # Demographics
        "age": int(age),
        "gender": gender_encoded,

        # Comorbidities / history
        "pmh_cirrhosis": int(cirrhosis == "Yes"),
        "pmh_fatty_liver": int(fatty_liver == "Yes"),
        "comorbid_diabetes": int(diabetes == "Yes"),
        "comorbid_htn": int(htn == "Yes"),
        "comorbid_cad": int(cad == "Yes"),

        # Systemic therapy
        "regimen_atezo_bev": regimen_atezo_bev,
        "regimen_durva_treme": regimen_durva_treme,
        "regimen_nivo_ipi": regimen_nivo_ipi,
        "regimen_pembro_ipi": regimen_pembro_ipi,

        # Local liver treatments
        "local_treatment_given_TACE": local_TACE,
        "local_treatment_given_Y90": local_Y90,
        "local_treatment_given_RFA": local_RFA,
        "local_treatment_given_None": local_None,

        # Other treatments
        "neoadjuvant_therapy": int(neoadjuvant == "Yes"),
        "adjuvant_treatment_given": int(adjuvant == "Yes"),

        # NER flags
        "liver_tumor_flag": int(liver_tumor_flag),
        "liver_disease_flag": int(liver_disease_flag),
        "portal_hypertension_flag": int(portal_hypertension_flag),
        "biliary_flag": int(biliary_flag),
        "symptoms_flag": int(symptoms_flag),

        # Model results
        "prediction": result["prediction"],
        "probability": round(result["probability"], 2)
    }


if st.session_state.has_prediction:
    st.markdown("---")
    st.subheader("Save patient data and model predictions to database")
    if st.button("Save Prediction to DB"):
        with st.spinner("Saving to database..."):
            try:
                r_db = requests.post(
                    "http://localhost:8000/api/v1/insert_data",
                    json=db_payload,
                    timeout=10
                )
                if r_db.status_code == 200:
                    st.success("✅ Prediction saved to database successfully")
                else:
                    st.error(f"❌ Failed to save: {r_db.status_code} - {r_db.text}")
            except Exception as e:
                st.error(f"❌ Error sending data: {e}")
    



    



