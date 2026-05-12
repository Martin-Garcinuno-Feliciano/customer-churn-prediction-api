from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field


MODEL_PATH = Path("models/churn_model.pkl")


class CustomerData(BaseModel):
    gender: str
    seniorcitizen: int = Field(ge=0, le=1)
    partner: str
    dependents: str
    tenure: int = Field(ge=0)
    phoneservice: str
    multiplelines: str
    internetservice: str
    onlinesecurity: str
    onlinebackup: str
    deviceprotection: str
    techsupport: str
    streamingtv: str
    streamingmovies: str
    contract: str
    paperlessbilling: str
    paymentmethod: str
    monthlycharges: float = Field(ge=0)
    totalcharges: float = Field(ge=0)


app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting customer churn probability using a trained machine learning model.",
    version="1.0.0",
)


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run python src/train_model.py first."
        )
    return joblib.load(MODEL_PATH)


model = load_model()


@app.get("/health")
def health_check() -> dict:
    return {
        "status": "ok",
        "model_loaded": model is not None,
    }


@app.post("/predict")
def predict(customer: CustomerData) -> dict:
    input_df = pd.DataFrame([customer.model_dump()])

    churn_probability = model.predict_proba(input_df)[0][1]
    prediction = model.predict(input_df)[0]

    return {
        "prediction": "Churn" if prediction == 1 else "No Churn",
        "churn_probability": round(float(churn_probability), 4),
    }