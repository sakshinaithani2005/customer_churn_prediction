# Customer Churn Prediction — XGBoost + DVC + SHAP + Docker

An end-to-end customer churn prediction pipeline using XGBoost, DVC, SHAP, Flask, Docker, and GitHub Actions.

This project trains an XGBoost classifier, evaluates its performance, explains feature importance and model decisions with SHAP, and serves the finalized model via a Dockerized Flask API.

---

## Architecture

```
                         GitHub
                            |
                            v
                    GitHub Actions
                            |
                            v
                           DVC
                            |
              +-------------+-------------+
              |                           |
              v                           v
       Data Ingestion              Preprocessing
            |                           |
            +-------------+-------------+
                          |
                          v
                       XGBoost
                          |
                          v
                  xgb_model.pkl
                          |
            +-------------+-------------+
            |                           |
            v                           v
       Evaluation                     SHAP
            |                           |
            v                           v
     Metrics/Threshold          SHAP Reports
            |                           |
            +-------------+-------------+
                          |
                          v
                  Deployment Artifacts
               model + encoder + threshold
                          |
                          v
                       Docker
                          |
                          v
                      Flask API
                          |
                          v
                    POST /predict
```

### Tool Responsibilities

| Tool | Purpose |
| :--- | :--- |
| **Git** | Version control |
| **DVC** | ML pipeline reproducibility and tracking |
| **XGBoost** | Churn prediction classification model |
| **SHAP** | Model explainability and feature impact |
| **Flask** | Serving predictions via REST API |
| **Docker** | Packaging and running the trained model consistently |
| **GitHub Actions** | CI/CD automation |

---

## Project Structure

```text
customer_churn_prediction/
│
├── .github/workflows/
│   └── ci.yml
├── data/
│   ├── raw/
│   ├── interim/
│   │   ├── encoder.pkl
│   │   ├── feature_names.txt
│   │   ├── x_train_processed.csv
│   │   └── x_test_processed.csv
│   └── output/
│       ├── xgb_model.pkl
│       └── model_metadata.json
├── reports/
│   ├── metrics.json
│   ├── predictions.csv
│   ├── threshold.txt
│   ├── threshold_results.csv
│   ├── confusion_matrix.png
│   ├── shap_summary.png
│   ├── shap_feature_importance.png
│   ├── shap_feature_importance.csv
│   └── shap_dependence_*.png
├── src/
│   ├── data_ingestion.py
│   ├── preprocessing.py
│   ├── model.py
│   ├── evaluate.py
│   └── shap_explain.py
├── app.py
├── Churn_Modelling.csv
├── dvc.yaml
├── dvc.lock
├── Dockerfile
├── .dockerignore
├── requirements.txt
├── requirements-api.txt
└── README.md
```

---

## Setup and Installation

### 1. Clone the Repository
```bash
git clone https://github.com/sakshinaithani2005/customer_churn_prediction
cd customer_churn_prediction
```

### 2. Create Virtual Environment

**Linux / WSL:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows PowerShell:**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Verify the installation of core ML packages:
```bash
python -c "import pandas; print('pandas OK')"
python -c "import sklearn; print('scikit-learn OK')"
python -c "import xgboost; print('xgboost OK')"
python -c "import shap; print('SHAP OK')"
```

---

## Data Version Control (DVC) Pipeline

If DVC has not yet been initialized in your local workspace:
```bash
dvc init
```

Verify DVC is working and inspect the pipeline layout:
```bash
dvc --version
dvc status
dvc dag
```

### Run the Complete Pipeline
```bash
dvc repro
```

The pipeline stages execute in the following order:
```text
data_ingestion -> preprocessing -> XGBoost model -> evaluation -> SHAP
```

DVC monitors dependencies (code, config, data files) and automatically skips stages whose inputs have not changed.

### Useful DVC Commands
* Initialize: `dvc init`
* Check status: `dvc status`
* Inspect DAG: `dvc dag`
* Run pipeline: `dvc repro`
* Track a manual file: `dvc add <file-or-directory>`
* Push/pull remote data: `dvc push` / `dvc pull`

> [!NOTE]
> `dvc push` and `dvc pull` require a configured remote storage location (e.g., S3, GCS, Azure Blob, or local directory).

---

## Running Individual Scripts

For debugging, you can run individual stages manually:
```bash
python src/data_ingestion.py
python src/preprocessing.py
python src/model.py
python src/evaluate.py
python src/shap_explain.py
```

For general work, always prefer `dvc repro` to ensure pipeline consistency.

---

## Model and Reports

The trained model is stored at:
```text
data/output/xgb_model.pkl
```

Other pipeline artifacts:
* Model Metadata: `data/output/model_metadata.json`
* Preprocessing Encoders: `data/interim/encoder.pkl`
* Registered Features: `data/interim/feature_names.txt`
* Performance metrics: `reports/metrics.json`
* Test Predictions: `reports/predictions.csv`
* Threshold Tuning Details: `reports/threshold.txt` & `reports/threshold_results.csv`
* Visualizations: `reports/confusion_matrix.png`

### Evaluation Metrics
The model is evaluated using the following:
* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* Confusion Matrix

The classification threshold is dynamically tuned on validation data and saved in `reports/threshold.txt`.

---

## Model Explainability (SHAP)

SHAP is integrated to interpret model decisions and identify feature contributions.

### Generated Reports
* `reports/shap_summary.png` (Overall summary beeswarm plot)
* `reports/shap_feature_importance.png` (Global feature importance bar chart)
* `reports/shap_feature_importance.csv` (Raw feature impact scores)
* `reports/shap_dependence_*.png` (Individual feature dependence plots)

### Key Features Identified
* Age
* NumOfProducts
* Gender_Male
* Geography_Germany
* ActiveAge (Derived feature)
* IsActiveMember
* ActiveTenure (Derived feature)
* Balance
* ZeroBalance (Derived feature)
* BalancePerProduct (Derived feature)

---

## Containerization (Docker)

To separate training from serving:
* **Training and Evaluation (DVC):** Handles ingestion, engineering, model training, SHAP, and metrics.
* **Serving (Docker & Flask):** Packages the runtime, dependencies, trained model artifacts, and Flask API.

The Docker container loads the pre-trained model and does not retrain.

### Build the Docker Image
```bash
docker build -t customer-churn-api:latest .
```

To perform a clean build ignoring cache:
```bash
docker build --no-cache -t customer-churn-api:latest .
```

Confirm the image is built successfully:
```bash
docker images
```

### Run the Container
```bash
docker run --rm -p 5000:5000 customer-churn-api:latest
```

The container starts a Flask server listening on port `5000` and automatically loads the model pipelines:
* `xgb_model.pkl`
* `encoder.pkl`
* `feature_names.txt`
* `threshold.txt`

---

## API Documentation and Testing

### 1. Health Check
```bash
curl http://localhost:5000
```
Expected response:
```json
{
  "message": "Customer Churn Prediction API",
  "model": "XGBoost",
  "status": "running",
  "threshold": 0.45
}
```

### 2. Predict Customer Churn
```bash
curl -X POST http://localhost:5000/predict \
-H "Content-Type: application/json" \
-d '{
    "CreditScore": 650,
    "Geography": "Germany",
    "Gender": "Female",
    "Age": 45,
    "Tenure": 5,
    "Balance": 100000,
    "NumOfProducts": 2,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 60000
}'
```

Expected response:
```json
{
  "churn": "No",
  "churn_probability": 0.4169,
  "prediction": 0,
  "threshold": 0.45
}
```

#### Threshold Logic
```text
If churn_probability < threshold (e.g. 0.4169 < 0.45):
    prediction = 0 (churn = No)
Else:
    prediction = 1 (churn = Yes)
```

---

## Reference Guides

### Docker CLI Cheat Sheet
* List running containers: `docker ps`
* List all containers: `docker ps -a`
* View logs: `docker logs <container_id>`
* Stop container: `docker stop <container_id>`
* Remove container: `docker rm <container_id>`
* Remove image: `docker rmi customer-churn-api:latest`
* Run interactive shell: `docker run -it customer-churn-api:latest bash`

---

## Continuous Integration (CI)

A GitHub Actions workflow is defined in `.github/workflows/ci.yml`.

### Workflow Steps
1. Push/Pull Request triggers the run.
2. Sets up Python and installs project dependencies.
3. Runs the DVC pipeline to verify reproducibility.
4. Asserts that model artifacts are created successfully.
5. Builds the Docker image.
6. Starts the container and tests the API response.

### Git Integration
To commit updates to the DVC pipeline and push them:
```bash
git add dvc.yaml dvc.lock
git commit -m "Update DVC pipeline tracking"
git push
```

---

## Typical Daily Development Workflow

1. Navigate to the project directory:
   ```bash
   cd customer_churn_prediction
   source .venv/bin/activate
   ```
2. Pull latest changes:
   ```bash
   git pull
   ```
3. Check status and run pipeline:
   ```bash
   dvc status
   dvc repro
   ```
4. Build and test container locally:
   ```bash
   docker build -t customer-churn-api:latest .
   docker run --rm -p 5000:5000 customer-churn-api:latest
   ```
5. Test API endpoints using `curl` or Postman.
6. Commit and push:
   ```bash
   git add .
   git commit -m "Update customer churn model pipeline"
   git push origin main
   ```