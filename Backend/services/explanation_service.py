import numpy as np

class ExplanationService:
    def __init__(self):
        # Full model features in training order (must match SHAP values)
        self.feature_names = [
            'ast', 'alt', 'alp', 'albumin', 'total_bilirubin', 'afp',
            'stage_at_diagnosis', 't_stage_at_diagnosis', 'age', 'gender',
            'pmh_cirrhosis', 'pmh_fatty_liver', 'comorbid_diabetes', 'comorbid_htn',
            'comorbid_cad', 'liver_tumor_flag', 'liver_disease_flag',
            'portal_hypertension_flag', 'biliary_flag', 'symptoms_flag',
            'regimen_atezo_bev', 'regimen_durva_treme', 'regimen_nivo_ipi',
            'regimen_pembro_ipi', 'local_treatment_given_TACE',
            'local_treatment_given_Y90', 'local_treatment_given_RFA',
            'local_treatment_given_None', 'neoadjuvant_therapy',
            'adjuvant_treatment_given'
        ]

        # Human-readable mapping
        self.feature_names_human = {
            'ast': "AST (liver enzyme)",
            'alt': "ALT (liver enzyme)",
            'alp': "ALP (liver enzyme)",
            'albumin': "Albumin level",
            'total_bilirubin': "Total bilirubin",
            'afp': "Alpha-fetoprotein (AFP)",
            'stage_at_diagnosis': "Cancer stage at diagnosis",
            't_stage_at_diagnosis': "Tumor stage at diagnosis",
            'age': "Patient age",
            'gender': "Patient gender",
            'pmh_cirrhosis': "History of cirrhosis",
            'pmh_fatty_liver': "History of fatty liver",
            'comorbid_diabetes': "Comorbidity: diabetes",
            'comorbid_htn': "Comorbidity: hypertension",
            'comorbid_cad': "Comorbidity: coronary artery disease",
            'liver_tumor_flag': "Presence of liver tumor",
            'liver_disease_flag': "Underlying liver disease",
            'portal_hypertension_flag': "Portal hypertension or complications",
            'biliary_flag': "Biliary/duct issues",
            'symptoms_flag': "Presence of symptoms",
            'regimen_atezo_bev': "Treatment: Atezolizumab + Bevacizumab",
            'regimen_durva_treme': "Treatment: Durvalumab + Tremelimumab",
            'regimen_nivo_ipi': "Treatment: Nivolumab + Ipilimumab",
            'regimen_pembro_ipi': "Treatment: Pembrolizumab + Ipilimumab",
            'local_treatment_given_TACE': "Local treatment: TACE",
            'local_treatment_given_Y90': "Local treatment: Y90",
            'local_treatment_given_RFA': "Local treatment: RFA",
            'local_treatment_given_None': "No local treatment",
            'neoadjuvant_therapy': "Neoadjuvant therapy given",
            'adjuvant_treatment_given': "Adjuvant therapy given"
        }

    def generate_explanation(self, probability: float, shap_values_list: list) -> str:
        """
        Generates a human-readable explanation based on SHAP values.

        Args:
            probability: Model-predicted probability for positive class
            shap_values_list: List or array of SHAP values (1D)

        Returns:
            A string explanation highlighting top positive and negative contributors
        """
        shap_values = np.array(shap_values_list).flatten()
        num_features = len(self.feature_names)

        # Ensure we do not exceed feature list length
        if len(shap_values) != num_features:
            print(f"Warning: SHAP values length ({len(shap_values)}) "
                  f"does not match number of features ({num_features}).")
            shap_values = shap_values[:num_features]

        # Get top 3 positive and negative contributors
        top_pos_idx = np.argsort(shap_values)[-3:][::-1]
        top_neg_idx = np.argsort(shap_values)[:3]

        top_pos_features = [self.feature_names[i] for i in top_pos_idx if i < num_features]
        top_neg_features = [self.feature_names[i] for i in top_neg_idx if i < num_features]

        # Convert to human-readable names
        pos_desc = ", ".join([self.feature_names_human.get(f, f) for f in top_pos_features])
        neg_desc = ", ".join([self.feature_names_human.get(f, f) for f in top_neg_features])

        explanation = (
            f"For this patient with hepatocellular carcinoma (HCC), the model predicts a probability of response "
            f"to the selected treatment regimen of {probability:.2f}. "
            f"The patient-specific attributes that contributed most positively to this prediction were: {pos_desc}. "
            f"The attributes that contributed most negatively were: {neg_desc}. "
            "Based on these factors, the model predicted whether the patient is likely to respond or not respond "
            "to this treatment, relative to the model's baseline prediction."
        )

        return explanation
        


