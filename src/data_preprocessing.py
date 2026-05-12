from pathlib import Path

import pandas as pd


RAW_DATA_PATH = Path("data/raw/telco_customer_churn.csv")
PROCESSED_DATA_PATH = Path("data/processed/telco_customer_churn_processed.csv")


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw Telco Customer Churn dataset."""
    if not path.exists():
        raise FileNotFoundError(f"Raw dataset not found at: {path}")
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw churn data and prepare target variable."""
    df = df.copy()

    # Standardize column names
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(" ", "_")
        .str.lower()
    )

    # Drop customer ID because it is an identifier, not a predictive feature
    if "customerid" in df.columns:
        df = df.drop(columns=["customerid"])

    # Convert TotalCharges from object to numeric.
    # Some rows contain blank spaces, which become NaN.
    df["totalcharges"] = pd.to_numeric(df["totalcharges"], errors="coerce")

    # Drop rows with missing TotalCharges.
    # These are usually customers with tenure = 0.
    df = df.dropna(subset=["totalcharges"])

    # Convert target variable to binary
    df["churn"] = df["churn"].map({"No": 0, "Yes": 1})

    return df


def save_processed_data(df: pd.DataFrame, path: Path = PROCESSED_DATA_PATH) -> None:
    """Save cleaned data to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    raw_df = load_raw_data()
    clean_df = clean_data(raw_df)
    save_processed_data(clean_df)

    print("Preprocessing complete.")
    print(f"Processed shape: {clean_df.shape}")
    print(f"Processed file saved to: {PROCESSED_DATA_PATH}")
    print("\nColumns:")
    for column in clean_df.columns:
        print(f"- {column}")


if __name__ == "__main__":
    main()