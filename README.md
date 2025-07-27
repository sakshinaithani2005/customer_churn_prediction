# Customer Churn Prediction 🚀

A machine learning project to predict whether a customer will churn based on historical data. This end-to-end pipeline uses **Random Forest Classifier**, **hyperparameter tuning**, **DVC** for experiment tracking, and **logging** for debugging and monitoring.

## 📌 Project Highlights

- 💡 **Model**: Random Forest Classifier  
- 🛠️ **Pipeline**: Built using DVC & YAML  
- 🔍 **Hyperparameter Tuning**: GridSearchCV  
- 🧪 **Logging**: Implemented using Python `logging` module  
- 📊 **Accuracy Achieved**: 85% on test data  

---


## ⚙️ Tools & Technologies Used

- **Python**
- **Scikit-learn**
- **Pandas & NumPy**
- **RandomForestClassifier**
- **GridSearchCV** (for hyperparameter tuning)
- **DVC** (for data and pipeline versioning)
- **YAML** (for defining pipeline stages and parameters)
- **Logging** (debug and error tracking)

---

## 🚀 Getting Started

1. Clone the repository
git clone https://github.com/yourusername/churn-prediction-dvc.git
cd churn-prediction-dvc
2. Set up virtual environment
bash
Copy
Edit
python -m venv .venv
source .venv/bin/activate   # for Unix
.venv\Scripts\activate      # for Windows
3. Install dependencies
bash
Copy
Edit
pip install -r requirements.txt
4. Reproduce pipeline
Make sure DVC is installed and then run:

bash
Copy
Edit
dvc repro
📈 Model Performance
Training Accuracy: ~85%

Evaluation Metrics: Precision, Recall, ROC-AUC (available in evaluate.py)

Logging: All events/errors are logged in logs/ directory.

🧪 Key Features
End-to-End Pipeline: From raw data preprocessing to model evaluation.

Reproducibility: Full reproducibility using DVC and versioned parameters.

Hyperparameter Optimization: Uses GridSearchCV to find the best Random Forest settings.

Modular Code: Easy to maintain and extend.

📌 Future Improvements
Add more models (e.g., XGBoost, SVM) for comparison

Implement model interpretability (e.g., SHAP or LIME)

Deploy the model using FastAPI or Streamlit

📝 License
This project is open-source under the MIT License.



