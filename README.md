# Customer Churn Prediction and Inference Gateway

An end-to-end machine learning system for predicting whether a bank customer is likely to churn. The project combines a reproducible XGBoost training pipeline, a Flask inference API, and an asynchronous reverse proxy that adds caching, rate limiting, health checks, metrics, and backend resilience.

## Problem Statement

Customer churn reduces revenue and increases the cost of acquiring replacement customers. A bank needs a reliable way to identify customers who may leave so that retention teams can prioritize timely, targeted interventions.

The challenge is to use customer profile and account information to estimate churn risk while also providing a service that can handle repeated prediction requests safely and efficiently.

## Solution

This project trains an XGBoost binary classification model on customer banking data. The solution:

- Cleans the source data and removes identifier columns.
- Splits the data into training, validation, and test sets.
- Engineers behavioral and financial features such as balance-to-salary ratio, active tenure, and credit-age interaction.
- Encodes categorical values consistently with the training process.
- Tunes and trains an XGBoost classifier.
- Evaluates the model and stores a validation-selected classification threshold.
- Exposes predictions through a Flask REST API.
- Places an asynchronous inference gateway in front of the API for caching, rate limiting, observability, and failure handling.

## What Is the Project About?

The project demonstrates the complete path from raw customer data to a production-oriented prediction service:

1. Data preparation and reproducible pipeline execution with DVC.
2. Feature engineering and model training with Python and scikit-learn-compatible tooling.
3. Model evaluation, threshold analysis, and SHAP-based feature importance.
4. REST deployment through Flask.
5. Request management through an `aiohttp` gateway.
6. Containerized operation with Docker Compose.

The model accepts ten customer attributes and returns a churn probability, a binary churn prediction, and the threshold used for that decision.

## Overall Working

```text
Churn_Modelling.csv
        |
        v
Data ingestion
  - clean identifiers
  - create train, validation, and test splits
        |
        v
Preprocessing
  - engineer features
  - one-hot encode Geography and Gender
  - save encoder and feature order
        |
        v
Model training
  - tune XGBoost with randomized search
  - save xgb_model.pkl and metadata
        |
        +--------------------+
        |                    |
        v                    v
Evaluation              SHAP analysis
- metrics               - feature importance
- threshold             - explanation reports
        |
        v
Flask API on port 5000
        |
        v
Inference gateway on port 8080
  - rate limiting
  - LRU cache with TTL
  - health endpoints
  - metrics
  - structured zero-PII logging
        |
        v
Client prediction response
```

For a `POST /predict` request, the gateway checks the client rate limit, creates a model-version-aware cache key, and returns a cached response when possible. Cache misses are forwarded to Flask. The Flask service recreates the training features, applies the saved encoder and feature order, calculates the churn probability, and applies the selected threshold. Successful responses are cached by the gateway.

When the Flask backend is unavailable, the gateway reports the backend failure and returns `503` for uncached predictions while continuing to serve valid cached predictions.

## Tools and Technologies

| Area | Tools |
|---|---|
| Language | Python 3.11 or newer |
| Data processing | pandas, NumPy |
| Machine learning | scikit-learn, XGBoost |
| Explainability | SHAP |
| API service | Flask |
| Inference gateway | aiohttp, asyncio |
| Model artifacts | joblib, JSON, CSV |
| Pipeline management | DVC |
| Testing and benchmarking | unittest, urllib, requests, custom load tests |
| Packaging and deployment | Docker, Docker Compose |
| Reporting | matplotlib, seaborn |

## Project Hierarchy

```text
customer_churn_prediction/
|-- app.py                         Flask prediction API
|-- Churn_Modelling.csv            Source customer dataset
|-- dvc.yaml                       Reproducible pipeline stages
|-- dvc.lock                       Locked DVC pipeline state
|-- Dockerfile                     Flask API image
|-- docker-compose.yml             Flask and gateway services
|-- requirements.txt               Training and analysis dependencies
|-- requirements-api.txt           Runtime API dependencies
|-- src/
|   |-- data_ingestion.py          Data cleaning and dataset splitting
|   |-- preprocessing.py           Feature engineering and encoding
|   |-- model.py                   XGBoost training and artifact creation
|   |-- evaluate.py                Metrics and threshold evaluation
|   `-- shap_explain.py            Model explainability reports
|-- data/
|   |-- raw/                       Train, validation, and test splits
|   |-- interim/                   Processed data, encoder, and feature names
|   `-- output/                    Trained model and metadata
|-- reports/                       Metrics, predictions, thresholds, and SHAP outputs
|-- proxy/
|   |-- gateway.py                 Async reverse proxy and inference gateway
|   |-- Dockerfile.proxy           Gateway image
|   |-- requirements.txt           Gateway dependency list
|   `-- tests/                     Unit, verification, and load tests
|-- logs/                          Pipeline logs
```

## Setup and Running the System

### Prerequisites

Install the following before starting:

- Python 3.11 or newer
- pip
- Docker and Docker Compose for containerized execution
- Git and DVC if you want to reproduce the pipeline from tracked data

All commands below should be run from the repository root.

### Option 1: Local Python Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
pip install -r proxy/requirements.txt
```

Generate or refresh the data and model artifacts:

```bash
dvc repro
```

The API requires these generated artifacts:

- `data/output/xgb_model.pkl`
- `data/interim/encoder.pkl`
- `data/interim/feature_names.txt`
- `reports/threshold.txt`

Start the Flask API in one terminal:

```bash
python app.py
```

Start the gateway in a second terminal:

```bash
python proxy/gateway.py
```

The services are available at:

- Flask API: `http://127.0.0.1:5000`
- Inference gateway: `http://127.0.0.1:8080`

### Option 2: Docker Compose

Build and start both services:

```bash
docker compose up --build
```

The gateway is exposed at `http://127.0.0.1:8080`. Stop the services with:

```bash
docker compose down
```

The Compose setup starts the gateway only after the Flask service passes its health check.

### API Endpoints

Gateway endpoints:

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Check that the gateway is running |
| `GET` | `/health/backend` | Check connectivity to the Flask backend |
| `GET` | `/metrics` | Read gateway request, cache, rate-limit, and latency metrics |
| `POST` | `/predict` | Request a churn prediction |

Example prediction request:

```bash
curl -X POST http://127.0.0.1:8080/predict \
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

The exact probability and threshold depend on the generated model artifacts.

### Tests and Benchmarking

Run the gateway unit tests:

```bash
python -m unittest discover -s proxy/tests -p "test_*.py"
```

Run the end-to-end gateway verification, including cache and backend failure checks:

```bash
python proxy/tests/verify_gateway.py
```

Run the load benchmark against running services:

```bash
python proxy/tests/load_test.py --requests 100 --concurrency 8
```

## Use Cases

- Prioritize customers for retention campaigns.
- Support relationship managers with account-level churn risk signals.
- Compare churn risk across customer segments and geographies.
- Trigger follow-up workflows in a customer engagement platform.
- Demonstrate production patterns for serving machine learning models.
- Measure the effect of response caching and rate limiting under repeated traffic.
- Investigate model behavior with SHAP feature importance reports.

Predictions should support customer service and retention decisions rather than replace human review. Model performance and fairness should be monitored before using the system for high-impact decisions.

## Future Work

- Add automated model and data validation checks before deployment.
- Track experiments, model versions, and metrics through a formal registry.
- Add calibration and segment-specific threshold analysis.
- Monitor data drift, prediction drift, latency, and error rates over time.
- Add authentication, authorization, TLS termination, and stronger request validation.
- Persist gateway metrics in a monitoring system such as Prometheus and Grafana.
- Add integration tests to the continuous integration pipeline.
- Improve cache sharing and invalidation for multi-instance deployments.
- Add batch prediction and asynchronous job support.
- Evaluate additional models and explainability methods.
- Add privacy controls, retention policies, and governance documentation for production customer data.
