from fastapi import FastAPI
from Backend.routes.predict import router as predict_router

app = FastAPI(
    title='HCC Prediction API',
    description='API for predicting HCC outcomes',
    version='1.0.0'
)

app.include_router(predict_router, prefix='/api/v1', tags=['prediction'])

@app.get('/')
def health_check():
    return {"status": "ok"}

