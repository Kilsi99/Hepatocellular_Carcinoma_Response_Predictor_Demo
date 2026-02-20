🧬 Hepatocellular Carcinoma Response Predictor
Overview

The Hepatocellular Carcinoma (HCC) Response Predictor is a full-stack clinical decision support application designed for clinicians and researchers.

It enables:

Prediction of treatment success probability for different HCC regimens

SHAP-based model explainability

Toxicity risk prediction across organ systems

Real-world outcome tracking and database storage

The system combines a FastAPI backend, a Streamlit frontend, and machine learning models (PyTorch / scikit-learn) to deliver interpretable clinical predictions.

🚀 Key Features
🔬 Clinical Prediction

Input patient laboratory values

Enter demographics and comorbidities

Provide tumor characteristics

Include free-text clinical notes

Generate probability of treatment success

Binary prediction (success / failure)

💡 Free-Text Clinical Note Processing (NER)

Demo Mode: For demonstration purposes, free-text notes are processed using a fake NER flag generator that randomly outputs the same binary flags as the real NER pipeline.

Example NER code (real pipeline):

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

model_name = "OpenMed/OpenMed-NER-PathologyDetect-PubMed-109M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
ner = pipeline("token-classification", model=model_name, tokenizer=tokenizer)

# Functions for extracting flags from clinical notes are implemented here...

This shows the real NER logic and model usage.
In production, the full model will be hosted and served via AWS to handle inference at scale.

Random flag generation for demo: The app currently generates fake flags to mimic NER output without loading heavy models. This keeps the demo lightweight.

📊 Model Explainability

SHAP-based feature contribution visualization

Transparent breakdown of key predictive variables

⚠️ Toxicity Risk Estimation

Predict toxicity probabilities per organ system

Supports risk-aware treatment planning

🗂 Outcome Tracking

Upload real-world treatment outcomes

Store predictions and outcomes in a backend database

Enable long-term performance evaluation

🏗 Architecture
Frontend (Streamlit)
        ↓
FastAPI Backend (REST API)
        ↓
Model Services (HCC + Toxicity + NER)
        ↓
Database Storage
Technology Stack

Frontend: Streamlit

Backend: FastAPI + Uvicorn

ML: PyTorch, scikit-learn

Explainability: SHAP

Deployment: Docker

Target Platform: Render (demo) / AWS (production for heavy NER model)

📦 Project Structure
HCC_APP_Demo/
│
├── Backend/
│   ├── main.py
│   ├── routes/
│   ├── services/
│   ├── schemas/
│   ├── requirements.txt
│   └── Dockerfile
│
├── Frontend/
│   ├── app.py
│   ├── requirements.txt
│   └── Dockerfile
│
├── models/        (ignored in git)
├── tests/
├── .gitignore
├── .dockerignore
└── README.md
🧪 Running Locally (Without Docker)
1️⃣ Backend
cd Backend
python -m venv .venv_backend
.\.venv_backend\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload

Backend will run at:

http://127.0.0.1:8000

API docs: http://127.0.0.1:8000/docs

2️⃣ Frontend

In a new terminal:

cd Frontend
python -m venv .venv_frontend
.\.venv_frontend\Scripts\activate
pip install -r requirements.txt
streamlit run app.py

Frontend runs at:

http://localhost:8501

🐳 Running with Docker

Build Backend:

docker build -t hcc-backend -f Backend/Dockerfile Backend/
docker run -p 8000:8000 hcc-backend

Build Frontend:

docker build -t hcc-frontend -f Frontend/Dockerfile Frontend/
docker run -p 8501:8501 hcc-frontend
🔐 Backend API

Core responsibilities:

Patient feature preprocessing

Model inference (HCC, toxicity, NER flags)

SHAP explanation generation

Database operations

Outcome persistence

All logic is handled via REST endpoints consumed by the Streamlit frontend.

📈 Machine Learning Components

HCC treatment success model

Toxicity prediction model

SHAP-based explanation service

Structured + free-text feature integration (NER)

🎯 Intended Users

Clinical researchers

Oncologists

Translational AI teams

Health data scientists

⚠️ Disclaimer

This application is a research and demonstration tool.
It is not intended for direct clinical decision-making without proper validation and regulatory approval.

👤 Author

Kilsi Kobani
MSc Health Data Science
Focused on clinical AI systems, interpretable ML, and deployable medical applications.