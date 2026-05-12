import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


PROCESSED_DATA_PATH = Path("data/processed/telco_customer_churn_processed.csv")
MODEL_PATH = Path("models/churn_model.pkl")
EVALUATION_PATH = Path("reports/evaluation_report.json")


def load_data(path: Path = PROCESSED_DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at: {path}. "
            "Run python src/data_preprocessing.py first."
        )

    return pd.read_csv(path)


def load_model(path: Path = MODEL_PATH):
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at: {path}. "
            "Run python src/train_model.py first."
        )

    return joblib.load(path)


def main() -> None:
    df = load_data()
    model = load_model()

    X = df.drop(columns=["churn"])
    y = df["churn"]

    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    cm = confusion_matrix(y, y_pred)

    report = {
        "note": (
            "This evaluation is run on the full processed dataset. "
            "For model selection, see reports/metrics.json, which uses a train/test split."
        ),
        "metrics": {
            "accuracy": round(accuracy_score(y, y_pred), 4),
            "precision": round(precision_score(y, y_pred), 4),
            "recall": round(recall_score(y, y_pred), 4),
            "f1_score": round(f1_score(y, y_pred), 4),
            "roc_auc": round(roc_auc_score(y, y_proba), 4),
        },
        "confusion_matrix": {
            "true_negative": int(cm[0][0]),
            "false_positive": int(cm[0][1]),
            "false_negative": int(cm[1][0]),
            "true_positive": int(cm[1][1]),
        },
        "classification_report": classification_report(
            y,
            y_pred,
            target_names=["No Churn", "Churn"],
            output_dict=True,
        ),
    }

    EVALUATION_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(EVALUATION_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print("Evaluation complete.")
    print(f"Evaluation report saved to: {EVALUATION_PATH}")
    print("\nMetrics:")
    print(json.dumps(report["metrics"], indent=4))
    print("\nConfusion matrix:")
    print(json.dumps(report["confusion_matrix"], indent=4))


if __name__ == "__main__":
    main()