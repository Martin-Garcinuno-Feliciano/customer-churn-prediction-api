from pathlib import Path

import pandas as pd


DATA_PATH = Path("data/raw/telco_customer_churn.csv")


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {DATA_PATH}. "
            "Place the Telco churn CSV in data/raw/ and rename it to telco_customer_churn.csv."
        )

    df = pd.read_csv(DATA_PATH)

    print("Dataset loaded successfully.")
    print(f"Shape: {df.shape}")
    print("\nColumns:")
    for column in df.columns:
        print(f"- {column}")

    print("\nFirst 5 rows:")
    print(df.head())


if __name__ == "__main__":
    main()