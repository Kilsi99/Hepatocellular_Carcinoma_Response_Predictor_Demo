import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from unittest.mock import MagicMock
import Backend.routes.predict as predict_routes
from Backend.main import app

client = TestClient(app)

def test_explanation_success():
    fake_results = {
        "probability": 0.85,
        "shap_values": [
            0.04783690784924375, -0.059978794526379844, 0.01997749093619847, 
            -0.06387663348845807, 0.01299370064749641, 0.0042625696283849455, 
            0.020121606881069187, 0.0013782034193244982, -0.0972794180278541, 
            -0.00039306199515222874, 0.0034023919578218074, 2.4171549345646913e-05, 
            -0.001972635903556388, 0.0004218522222773541, -0.0019488898164254872, 
            -0.000138617939396385, -3.9403533927284305e-05, -1.3184698992732247e-05, 
            0.0002522692268142169, 0.005360969814176188, -0.0035942238646169264, 
            0.0037812613615177833, 0.0, 0.0, -0.00041584717675706336, 0.0018325593806573497, 
            0.014838789922513269, 0.001937969403829158, 0.004105804214865594, -0.0003699169539646131
        ]
    }

    response = client.post("/api/v1/explanation", json=fake_results)
    
    assert response.status_code == 200
    json_data = response.json()
    assert "explanation" in json_data
    print("Explanation response:", json_data)