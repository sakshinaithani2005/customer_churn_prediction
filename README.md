Customer Churn Prediction — XGBoost + DVC + SHAP + Docker

End-to-end customer churn prediction using XGBoost, DVC, SHAP, Flask, Docker, and GitHub Actions.

The project trains an XGBoost model, evaluates it, explains it with SHAP, and serves the already-trained model through a Dockerized Flask API.

Architecture

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

Tool responsibilities

Tool

Purpose

Git

Version control

DVC

ML pipeline and reproducibility

XGBoost

Churn prediction

SHAP

Model explainability

Flask

Prediction API

Docker

Package and run the trained model

GitHub Actions

CI/CD automation

Project Structure

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

1. Start From Scratch

Clone

git clone <YOUR_GITHUB_REPOSITORY_URL>
cd customer_churn_prediction

Create virtual environment

Linux / WSL:

python3 -m venv .venv
source .venv/bin/activate

Windows PowerShell:

python -m venv .venv
.venv\Scripts\Activate.ps1

Verify:

python --version

Install dependencies

python -m pip install --upgrade pip
pip install -r requirements.txt

Verify important packages:

python -c "import pandas; print('pandas OK')"
python -c "import sklearn; print('scikit-learn OK')"
python -c "import xgboost; print('xgboost OK')"
python -c "import shap; print('SHAP OK')"

2. DVC

If DVC has not already been initialized:

dvc init

Check:

dvc --version
dvc status
dvc dag

Run the complete pipeline

dvc repro

Pipeline:

data_ingestion
      ↓
preprocessing
      ↓
XGBoost model
      ↓
evaluation
      ↓
SHAP

DVC can skip stages whose dependencies have not changed.

Useful DVC commands

dvc init
dvc status
dvc dag
dvc repro
dvc add <file-or-directory>
dvc push
dvc pull
dvc config --list

dvc push and dvc pull require a configured DVC remote.

3. Run Individual Scripts

For debugging:

python src/data_ingestion.py
python src/preprocessing.py
python src/model.py
python src/evaluate.py
python src/shap_explain.py

For normal execution, prefer:

dvc repro

4. Model and Reports

The trained model is:

data/output/xgb_model.pkl

Other artifacts include:

data/output/model_metadata.json
data/interim/encoder.pkl
data/interim/feature_names.txt
reports/metrics.json
reports/predictions.csv
reports/threshold.txt
reports/threshold_results.csv
reports/confusion_matrix.png

Evaluation uses:

Accuracy

Precision

Recall

F1 Score

ROC-AUC

Confusion Matrix

The validation-selected classification threshold is stored in:

reports/threshold.txt

5. SHAP

SHAP explains the XGBoost model.

Outputs include:

reports/shap_summary.png
reports/shap_feature_importance.png
reports/shap_feature_importance.csv
reports/shap_dependence_*.png

Important features observed in the project included:

Age
NumOfProducts
Gender_Male
Geography_Germany
ActiveAge
IsActiveMember
ActiveTenure
Balance
ZeroBalance
BalancePerProduct

6. Docker

The final architecture separates training from serving.

DVC handles

Training
Evaluation
SHAP
Model artifacts

Docker handles

Runtime environment
Trained XGBoost model
Flask API
Prediction serving

The Docker container does not retrain the model.

7. Docker Setup

Check Docker:

docker --version
docker info

The API uses:

requirements-api.txt

It contains only dependencies needed for model serving.

8. Build Docker Image

From the project root:

docker build -t customer-churn-api:latest .

For a complete rebuild without cache:

docker build --no-cache -t customer-churn-api:latest .

Normally use the cached build:

docker build -t customer-churn-api:latest .

9. Check Docker Images

docker images

Expected image:

customer-churn-api    latest

10. Run Docker Container

docker run --rm -p 5000:5000 customer-churn-api:latest

The API runs at:

http://localhost:5000

The container loads:

xgb_model.pkl
encoder.pkl
feature_names.txt
threshold.txt

11. Test Flask API

Health check:

curl http://localhost:5000

Expected response is similar to:

{
  "message": "Customer Churn Prediction API",
  "model": "XGBoost",
  "status": "running",
  "threshold": 0.45
}

Prediction request

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

Example:

{
  "churn": "No",
  "churn_probability": 0.4169,
  "prediction": 0,
  "threshold": 0.45
}

Interpretation:

0.4169 < 0.45
       ↓
prediction = 0
       ↓
churn = No

12. Useful Docker Commands

List running containers:

docker ps

List all containers:

docker ps -a

View logs:

docker logs <container_id>

Stop:

docker stop <container_id>

Remove:

docker rm <container_id>

Remove image:

docker rmi customer-churn-api:latest

Open a shell inside the image:

docker run -it customer-churn-api:latest bash

13. GitHub Actions / CI

The CI workflow is under:

.github/workflows/

General flow:

Git Push / Pull Request
          ↓
    GitHub Actions
          ↓
 Install dependencies
          ↓
     DVC pipeline
          ↓
 Verify artifacts
          ↓
    Build Docker
          ↓
    Run container

Useful Git commands:

git status
git add .
git commit -m "Update customer churn pipeline"
git push origin main

After changing DVC pipeline files:

git add dvc.yaml dvc.lock
git commit -m "Update DVC pipeline"
git push

14. Typical Daily Workflow

cd customer_churn_prediction

source .venv/bin/activate

git pull

dvc status

dvc repro

Then build and run the API:

docker build -t customer-churn-api:latest .
docker run --rm -p 5000:5000 customer-churn-api:latest

Test:

curl http://localhost:5000

Then test /predict.

Finally:

git status
git add .
git commit -m "Update customer churn model"
git push origin main

15. Complete Project Flow

              Churn_Modelling.csv
                       |
                       v
              data_ingestion.py
                       |
                       v
                Train/Val/Test
                       |
                       v
               preprocessing.py
                       |
              Feature Engineering
                       +
                One-Hot Encoding
                       |
                       v
                    XGBoost
                       |
                       v
                xgb_model.pkl
                       |
             +---------+---------+
             |                   |
             v                   v
        evaluate.py       shap_explain.py
             |                   |
             v                   v
      Metrics/Threshold      SHAP Reports
             |
             v
      Deployment Artifacts
             |
             v
           Docker
             |
             v
        Flask API
             |
             v
        POST /predict
             |
             v
      Churn Probability
             |
             v
        Final Prediction

Quick Cheat Sheet

Environment

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

DVC

dvc init
dvc status
dvc dag
dvc repro
dvc add <file-or-directory>
dvc push
dvc pull

Training

python src/data_ingestion.py
python src/preprocessing.py
python src/model.py
python src/evaluate.py
python src/shap_explain.py

Docker

docker build -t customer-churn-api:latest .
docker images
docker run --rm -p 5000:5000 customer-churn-api:latest
docker ps
docker ps -a
docker logs <container_id>
docker stop <container_id>
docker rm <container_id>
docker rmi customer-churn-api:latest

API

curl http://localhost:5000

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

Project Summary

Data → Feature Engineering → XGBoost → Evaluation → SHAP → DVC → Docker → Flask API → CI/CD

DVC manages the reproducible ML pipeline. Docker packages the trained model and its serving environment. Flask exposes the trained XGBoost model through /predict.