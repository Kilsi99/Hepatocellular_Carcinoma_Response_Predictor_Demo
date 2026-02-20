from pydantic import BaseModel
from typing import List

class ResultsData(BaseModel):
    probability: float
    shap_values: List[float]





        