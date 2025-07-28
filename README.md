# Customer Churn Prediction

This project predicts customer churn using a machine learning pipeline. The pipeline is built using DVC and YAML configuration files. The model is trained using Random Forest Classifier and achieves 85% accuracy. The code includes logging for error tracking and hyperparameter tuning using GridSearchCV.

## Under Development
 So all feautres are not implemented yet

 
## Project Highlights

- Model: Random Forest Classifier  
- Pipeline: Built using DVC and YAML  
- Hyperparameter Tuning: Performed using GridSearchCV  
- Logging: Implemented with Python's logging module  
- Accuracy Achieved: 85% on test data  


## 🚀 Features

- 🔄 **Data Version Control (DVC)**: Track and version datasets and ML models.
- 🧹 **Data Preprocessing**: Clean, transform, and split raw data.
- 🧠 **ML Modeling**: Train and evaluate predictive models.
- 📦 **Model Registry**: Store and version trained models.
- 🧪 **Testing Suite**: Unit tests for pipeline components.
- ⚙️ **CI/CD Integration**: GitHub Actions for linting, testing, and model training.
- ☁️ **Cloud-ready**: Configurable for deployment on AWS, Azure, or GCP.

---

## 🗂️ Project Structure

```

retainStack/
│
├── data/                      # Raw and processed datasets
├── logger/
|   ├── logger.py              # Logger module for log monitoring
├── src/                       # Source code
│   ├── data_ingestion.py     # Ingestion scripts
│   ├── preprocessing.py  # Cleaning and splitting logic
│   ├── model.py            # Model Evaluation
│   
│  
├── dvc.yaml                   # DVC pipeline definition
├── params.yaml                # Hyperparameters & config
├── .github/workflows/         # CI/CD workflows
├── requirements.txt
├── README.md
└── .gitignore

````

---

## 📦 Setup Instructions

1. **Clone the repo**
   ```bash
   git clone https://github.com/sakshinaithani2005/customer_churn_prediction
   cd RetainStack
    ```

2. **Create virtual environment**

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Linux/Mac
   .venv\Scripts\activate      # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up DVC**

   ```bash
   dvc init
   dvc pull
   ```

---

## ⚙️ Running the Pipeline

To run the full DVC pipeline:

```bash
dvc repro
```

To run individual stages (e.g., data ingestion):

```bash
python src/data_ingestion.py
```

---

## 📁 Configuration

All configuration (paths, split ratios, model parameters) is defined in:

* `params.yaml` for hyperparameters
* `config.py` for directory structure
* `dvc.yaml` for pipeline stages

---


## 📌 Future Improvements

* Streamlit or FastAPI serving
* MLflow for model tracking
* Full cloud deployment (SageMaker, Vertex AI)

---

## 👨‍💻 Author

**Sakshi**
---

