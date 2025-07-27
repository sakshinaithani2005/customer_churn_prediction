import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import logging
import os

# --- Logging Setup ---
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('model.py')
logger.setLevel(logging.DEBUG)

if not logger.hasHandlers():
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(os.path.join(log_dir, 'model.log'))
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

# --- Model Training ---
def model(x_train, y_train, x_test):
    rf = RandomForestClassifier(random_state=42)

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2']
    }

    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring='accuracy',
        cv=5,
        verbose=1,
        n_jobs=-1
    )

    grid_search.fit(x_train, y_train)

    best_rf = grid_search.best_estimator_
    logger.info("Best Parameters: %s", grid_search.best_params_)

    y_pred = best_rf.predict(x_test)
    return y_pred

# --- Evaluation ---
def evaluate(y_test, y_pred):
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print("Test Accuracy:", acc)
    print("Classification Report:\n", report)
    logger.info("Model Evaluation:\nAccuracy: %.4f\n%s", acc, report)

# --- Main ---
def main():
    try:
        x_train = pd.read_csv('./data/interim/x_train_processed.csv')
        x_test = pd.read_csv('./data/interim/x_test_processed.csv')
        y_train = pd.read_csv('./data/raw/y_train.csv').values.ravel()
        y_test = pd.read_csv('./data/raw/y_test.csv').values.ravel()

        logger.debug('Data loaded successfully.')

        y_pred = model(x_train, y_train, x_test)

        evaluate(y_test, y_pred)

        output_dir = "./data/output"
        os.makedirs(output_dir, exist_ok=True)
        pd.DataFrame(y_pred, columns=["Prediction"]).to_csv(os.path.join(output_dir, "y_pred.csv"), index=False)

        logger.debug('Model ran and predictions saved successfully.')

    except FileNotFoundError as e:
        logger.error('File not found: %s', e)
    except Exception as e:
        logger.error('An unexpected error occurred: %s', e)

if __name__ == "__main__":
    main()
