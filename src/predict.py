from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = Path("models/churn_model.pkl")


def load_model(path: Path = MODEL_PATH):
    """Load the trained churn prediction model."""
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at: {path}. Run python src/train_model.py first."
        )

    return joblib.load(path)


def predict_churn(customer_data: dict) -> dict:
    """Predict churn probability for a single customer."""
    model = load_model()

    input_df = pd.DataFrame([customer_data])

    churn_probability = model.predict_proba(input_df)[0][1]
    prediction = model.predict(input_df)[0]

    return {
        "prediction": "Churn" if prediction == 1 else "No Churn",
        "churn_probability": round(float(churn_probability), 4),
    }


def main() -> None:
    sample_customer = {
        "gender": "Female",
        "seniorcitizen": 0,
        "partner": "Yes",
        "dependents": "No",
        "tenure": 12,
        "phoneservice": "Yes",
        "multiplelines": "No",
        "internetservice": "Fiber optic",
        "onlinesecurity": "No",
        "onlinebackup": "Yes",
        "deviceprotection": "No",
        "techsupport": "No",
        "streamingtv": "Yes",
        "streamingmovies": "Yes",
        "contract": "Month-to-month",
        "paperlessbilling": "Yes",
        "paymentmethod": "Electronic check",
        "monthlycharges": 75.3,
        "totalcharges": 900.5,
    }

    result = predict_churn(sample_customer)

    print("Sample prediction:")
    print(result)


if __name__ == "__main__":
    main()