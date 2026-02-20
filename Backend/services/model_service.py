# Backend/services/model_service.py

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.preprocessing import StandardScaler

# Feature names must match build_features order
FEATURE_NAMES = [
    'ast', 'alt', 'alp', 'albumin', 'total_bilirubin', 'afp',
    'stage_at_diagnosis', 't_stage_at_diagnosis', 'age', 'gender',
    'pmh_cirrhosis', 'pmh_fatty_liver', 'comorbid_diabetes',
    'comorbid_htn', 'comorbid_cad',
    'liver_tumor_flag', 'liver_disease_flag', 'portal_hypertension_flag',
    'biliary_flag', 'symptoms_flag',
    'regimen_atezo_bev', 'regimen_durva_treme',
    'regimen_nivo_ipi', 'regimen_pembro_ipi',
    'local_treatment_given_TACE', 'local_treatment_given_Y90',
    'local_treatment_given_RFA', 'local_treatment_given_None',
    'neoadjuvant_therapy', 'adjuvant_treatment_given'
]

# -----------------------------
# HCC Model Service
# -----------------------------
class HCCModelService:
    def __init__(self, model_path: str):
        """
        Load the HCC model (RandomForest or LogisticRegression) and setup SHAP explainer.
        """
        self.model = joblib.load(model_path)
        
        # Use TreeExplainer for RandomForest, LinearExplainer for LogisticRegression
        try:
            self.explainer = shap.TreeExplainer(self.model)
            self.model_type = "tree"
        except:
            self.explainer = shap.LinearExplainer(self.model, np.zeros((1, len(FEATURE_NAMES))))
            self.model_type = "linear"

    def _to_df(self, X: np.ndarray) -> pd.DataFrame:
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return pd.DataFrame(X, columns=FEATURE_NAMES)

    def hcc_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Returns probability array (n_samples, 2) for binary classification
        """
        X_df = self._to_df(X)
        return self.model.predict_proba(X_df)

    def explain_prediction(self, X: np.ndarray):
        """
        Returns SHAP values and base value for the first sample
        """
        X_df = self._to_df(X)
        shap_values = self.explainer.shap_values(X_df)
        if self.model_type == "tree":
            base_value = self.explainer.expected_value[1]  # positive class
            return shap_values[0][:, 1], base_value
        else:
            base_value = self.explainer.expected_value[1] if hasattr(self.explainer, 'expected_value') else 0
            return shap_values[0], base_value

# -----------------------------
# Toxicity Model Service
# -----------------------------
class ToxicityModelService:
    def __init__(self, model_path: str, scaler_path: str = None):
        """
        Load a logistic regression model for multi-organ toxicity prediction.
        Optional scaler can be applied if features were standardized.
        """
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path) if scaler_path else None

    def _scale(self, X: np.ndarray) -> np.ndarray:
        if self.scaler:
            return self.scaler.transform(X)
        return X

    def predict_toxicity(self, X: np.ndarray) -> np.ndarray:
        """
        Returns probability array (n_samples, n_targets)
        """
        X_scaled = self._scale(X)
        return self.model.predict_proba(X_scaled)  # returns list of arrays if multi-class
