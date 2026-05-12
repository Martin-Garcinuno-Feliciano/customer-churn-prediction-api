# Customer Churn Prediction API

## Project Overview

This project builds a machine learning API to predict customer churn for a telecom company.

The goal is to create an end-to-end AI engineering workflow: data preprocessing, model training, model evaluation, model persistence, API development, and Docker containerization.

The final system receives customer information as JSON and returns a churn prediction with a probability score.

## Tech Stack

- Python
- pandas
- scikit-learn
- FastAPI
- Pydantic
- Uvicorn
- Docker
- joblib

## Dataset

This project uses the Telco Customer Churn dataset.

The dataset contains customer-level information such as:

- Contract type
- Tenure
- Monthly charges
- Total charges
- Internet service
- Tech support
- Payment method
- Churn status

The target variable is:

```text
Churn
```

Where:

```text
No  → 0
Yes → 1
```

## Project Structure

```text
customer-churn-prediction-api/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   └── churn_model.pkl
│
├── notebooks/
│
├── reports/
│   ├── metrics.json
│   └── evaluation_report.json
│
├── src/
│   ├── api.py
│   ├── data_preprocessing.py
│   ├── evaluate_model.py
│   ├── predict.py
│   └── train_model.py
│
├── Dockerfile
├── .dockerignore
├── .gitignore
├── requirements.txt
└── README.md
```

## Workflow

### 1. Data Preprocessing

Run:

```bash
python src/data_preprocessing.py
```

This script:

- Loads the raw dataset
- Cleans column names
- Drops the customer ID column
- Converts `TotalCharges` to numeric
- Removes missing values
- Converts the target variable `Churn` to binary format
- Saves the processed dataset

Output:

```text
data/processed/telco_customer_churn_processed.csv
```

The raw dataset contains 7,043 rows and 21 columns. After preprocessing, the processed dataset contains 7,032 rows and 20 columns.

### 2. Model Training

Run:

```bash
python src/train_model.py
```

The training script compares three classification models:

- Logistic Regression
- Random Forest
- Gradient Boosting

The best model is selected using ROC-AUC.

Outputs:

```text
models/churn_model.pkl
reports/metrics.json
```

### 3. Model Evaluation

Run:

```bash
python src/evaluate_model.py
```

This generates a structured evaluation report:

```text
reports/evaluation_report.json
```

The report includes:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix
- Classification report

## Model Results

The best model selected by ROC-AUC was:

```text
Gradient Boosting
```

Train/test evaluation results:

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7257 | 0.4901 | 0.7968 | 0.6069 | 0.8351 |
| Random Forest | 0.7839 | 0.6224 | 0.4759 | 0.5394 | 0.8116 |
| Gradient Boosting | 0.7960 | 0.6408 | 0.5294 | 0.5798 | 0.8385 |

Gradient Boosting achieved the best ROC-AUC, while Logistic Regression achieved the highest recall.

In a real churn prevention use case, the final model choice would depend on the business objective. If the business wants to catch as many potential churners as possible, recall should be prioritized. If the business wants fewer false churn alerts, precision should be prioritized.

## Full-Dataset Evaluation

The saved model was also evaluated on the full processed dataset to generate an additional report.

| Metric | Value |
|---|---:|
| Accuracy | 0.8234 |
| Precision | 0.7067 |
| Recall | 0.5736 |
| F1-score | 0.6332 |
| ROC-AUC | 0.8742 |

Confusion matrix:

| Result | Count |
|---|---:|
| True Negative | 4,718 |
| False Positive | 445 |
| False Negative | 797 |
| True Positive | 1,072 |

Note: the train/test results in `reports/metrics.json` are the main results used for model selection. The full-dataset evaluation is included as an additional diagnostic report.

## API Usage

Run the API locally:

```bash
uvicorn src.api:app --reload
```

Open the interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

### Health Check

Endpoint:

```http
GET /health
```

Example response:

```json
{
  "status": "ok",
  "model_loaded": true
}
```

### Churn Prediction

Endpoint:

```http
POST /predict
```

Example request:

```json
{
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
  "totalcharges": 900.5
}
```

Example response:

```json
{
  "prediction": "Churn",
  "churn_probability": 0.5747
}
```

## Docker Usage

Build the Docker image:

```bash
docker build -t churn-api .
```

Run the container:

```bash
docker run -p 8000:8000 churn-api
```

Then open:

```text
http://127.0.0.1:8000/docs
```

The Dockerized API exposes the same endpoints:

```text
GET /health
POST /predict
```

## Business Interpretation

The model predicts the probability that a telecom customer will churn based on customer profile, contract, billing, and service-related variables.

This type of system could support a customer retention team by identifying customers with high churn risk. The business could then prioritize retention campaigns, targeted offers, or customer support interventions for the highest-risk customers.

The project also highlights an important business tradeoff:

- A high-recall model catches more potential churners but may generate more false positives.
- A high-precision model creates fewer false churn alerts but may miss more actual churners.

For a real deployment, the classification threshold should be tuned according to the cost of false positives versus false negatives.

## Future Improvements

Potential extensions:

- Tune the classification threshold to improve recall
- Add MLflow experiment tracking
- Add automated API tests
- Add SHAP model explainability
- Add a Streamlit frontend
- Deploy the API to a cloud platform