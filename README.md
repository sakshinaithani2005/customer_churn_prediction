# Customer Churn Prediction

This project predicts whether a bank customer is likely to churn using a trained XGBoost model. It covers the full machine learning workflow from raw customer data to a usable API for inference.

## Overview

Customer churn reduces revenue and increases the cost of acquiring replacement customers. This project helps a bank identify at-risk customers early so retention teams can act proactively.

The project includes:
- data ingestion and cleaning
- feature engineering and preprocessing
- model training with XGBoost
- evaluation with metrics and threshold tuning
- SHAP-based explainability analysis
- a Flask REST API for inference

## Project Structure

```text
customer_churn_prediction/
|-- app.py                         Flask prediction API
|-- Churn_Modelling.csv            Source customer dataset
|-- dvc.yaml                       Reproducible pipeline stages
|-- dvc.lock                       Locked DVC pipeline state
|-- requirements.txt               Training and analysis dependencies
|-- requirements-api.txt           Runtime API dependencies
|-- src/
|   |-- data_ingestion.py          Data cleaning and dataset splitting
|   |-- preprocessing.py           Feature engineering and encoding
|   |-- model.py                   XGBoost training and artifact creation
|   |-- evaluate.py                Metrics and threshold evaluation
|   `-- shap_explain.py            Explainability reports
|-- data/
|   |-- raw/                       Train, validation, and test splits
|   |-- interim/                   Processed data, encoder, and feature names
|   `-- output/                    Trained model and metadata
|-- reports/                       Metrics, predictions, thresholds, outputs
|-- logs/                          Pipeline logs
```

## Prerequisites

- Python 3.11 or newer
- pip
- Git and DVC if you want to reproduce the training pipeline

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the project dependencies:

```bash
pip install -r requirements.txt
```

Generate the training artifacts and model outputs:

```bash
dvc repro
```

The API requires these generated files:

- `data/output/xgb_model.pkl`
- `data/interim/encoder.pkl`
- `data/interim/feature_names.txt`
- `reports/threshold.txt`

## Running the API

Start the Flask service:

```bash
python app.py
```

The API will be available at:

- `http://127.0.0.1:5000`

## API Endpoints

### `GET /`
Returns basic service information and the current model threshold.

### `POST /predict`
Accepts a JSON payload with the required customer attributes and returns the churn probability and prediction.

Required fields:
- `CreditScore`
- `Geography`
- `Gender`
- `Age`
- `Tenure`
- `Balance`
- `NumOfProducts`
- `HasCrCard`
- `IsActiveMember`
- `EstimatedSalary`

Example request:

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "CreditScore": 650,
    "Geography": "Germany",
    "Gender": "Female",
    "Age": 45,
    "Tenure": 5,
    "Balance": 100000.0,
    "NumOfProducts": 2,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 60000.0
  }'
```

Example response:

```json
{
  "churn_probability": 0.42,
  "prediction": 0,
  "churn": "No",
  "threshold": 0.5
}
```

## Reproducible ML Pipeline

The training pipeline is defined in `dvc.yaml` and can be reproduced with:

```bash
dvc repro
```

This workflow prepares the data, trains the model, evaluates it, and produces the artifacts used by the API.

## Notes

This project is focused on direct API-based churn prediction inference and model deployment.
