import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROCESSED_DATA_PATH = Path("data/processed/telco_customer_churn_processed.csv")
MODEL_PATH = Path("models/churn_model.pkl")
METRICS_PATH = Path("reports/metrics.json")


def load_data(path: Path = PROCESSED_DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at: {path}. "
            "Run python src/data_preprocessing.py first."
        )
    return pd.read_csv(path)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor


def evaluate_classifier(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
    }


def main() -> None:
    df = load_data()

    X = df.drop(columns=["churn"])
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    preprocessor = build_preprocessor(X_train)

    candidates = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=42),
    }

    results = {}
    trained_models = {}

    for model_name, classifier in candidates.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", classifier),
            ]
        )

        pipeline.fit(X_train, y_train)
        metrics = evaluate_classifier(pipeline, X_test, y_test)

        results[model_name] = metrics
        trained_models[model_name] = pipeline

    # Select best model by ROC-AUC.
    # For churn use cases, recall is also important, but ROC-AUC is a stable general ranking metric.
    best_model_name = max(results, key=lambda name: results[name]["roc_auc"])
    best_model = trained_models[best_model_name]

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_model, MODEL_PATH)

    output = {
        "best_model": best_model_name,
        "model_selection_metric": "roc_auc",
        "results": results,
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)

    print("Training complete.")
    print(f"Best model: {best_model_name}")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")
    print("\nResults:")
    print(json.dumps(output, indent=4))


if __name__ == "__main__":
    main()