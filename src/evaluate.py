import pandas as pd
import numpy as np
import logging
import os
import json
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = "data/output/xgb_model.pkl"

INTERIM_DIR = "data/interim"
RAW_DIR = "data/raw"

REPORT_DIR = "reports"
LOG_DIR = "logs"

# Thresholds used for validation threshold tuning
THRESHOLDS = np.arange(
    0.20,
    0.71,
    0.01
)


# ============================================================
# Logging Setup
# ============================================================

os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("evaluation")
logger.setLevel(logging.DEBUG)

# Prevent duplicate handlers
if not logger.handlers:

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    file_handler = logging.FileHandler(
        os.path.join(
            LOG_DIR,
            "evaluation.log"
        )
    )
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - "
        "%(levelname)s - %(message)s"
    )

    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


# ============================================================
# Load Model
# ============================================================

def load_model():

    logger.info(
        "Loading trained XGBoost model..."
    )

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            f"Model not found at: {MODEL_PATH}"
        )

    model = joblib.load(
        MODEL_PATH
    )

    logger.info(
        "XGBoost model loaded successfully."
    )

    return model


# ============================================================
# Load Validation Data
# ============================================================

def load_validation_data():

    logger.info(
        "Loading validation data..."
    )

    x_val_path = os.path.join(
        INTERIM_DIR,
        "x_val_processed.csv"
    )

    y_val_path = os.path.join(
        RAW_DIR,
        "y_val.csv"
    )

    if not os.path.exists(x_val_path):
        raise FileNotFoundError(
            f"Validation features not found: {x_val_path}"
        )

    if not os.path.exists(y_val_path):
        raise FileNotFoundError(
            f"Validation target not found: {y_val_path}"
        )

    x_val = pd.read_csv(
        x_val_path
    )

    y_val = pd.read_csv(
        y_val_path
    ).values.ravel()

    logger.info(
        "Validation data shape: %s",
        x_val.shape
    )

    logger.info(
        "Validation target shape: %s",
        y_val.shape
    )

    return x_val, y_val


# ============================================================
# Load Test Data
# ============================================================

def load_test_data():

    logger.info(
        "Loading test data..."
    )

    x_test_path = os.path.join(
        INTERIM_DIR,
        "x_test_processed.csv"
    )

    y_test_path = os.path.join(
        RAW_DIR,
        "y_test.csv"
    )

    if not os.path.exists(x_test_path):
        raise FileNotFoundError(
            f"Test features not found: {x_test_path}"
        )

    if not os.path.exists(y_test_path):
        raise FileNotFoundError(
            f"Test target not found: {y_test_path}"
        )

    x_test = pd.read_csv(
        x_test_path
    )

    y_test = pd.read_csv(
        y_test_path
    ).values.ravel()

    logger.info(
        "Test data shape: %s",
        x_test.shape
    )

    logger.info(
        "Test target shape: %s",
        y_test.shape
    )

    return x_test, y_test


# ============================================================
# Find Best Threshold
# ============================================================

def find_best_threshold(
    y_true,
    y_prob
):

    logger.info(
        "Searching for the best classification threshold..."
    )

    threshold_results = []

    for threshold in THRESHOLDS:

        # Convert probabilities into predictions
        y_pred = (
            y_prob >= threshold
        ).astype(int)

        precision = precision_score(
            y_true,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_true,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_true,
            y_pred,
            zero_division=0
        )

        accuracy = accuracy_score(
            y_true,
            y_pred
        )

        threshold_results.append({

            "threshold": float(
                threshold
            ),

            "accuracy": float(
                accuracy
            ),

            "precision": float(
                precision
            ),

            "recall": float(
                recall
            ),

            "f1": float(
                f1
            )
        })

    threshold_df = pd.DataFrame(
        threshold_results
    )

    # Select threshold based ONLY on validation F1
    best_row = threshold_df.loc[
        threshold_df["f1"].idxmax()
    ]

    best_threshold = float(
        best_row["threshold"]
    )

    logger.info(
        "Best validation threshold: %.2f",
        best_threshold
    )

    logger.info(
        "Validation Precision: %.4f",
        best_row["precision"]
    )

    logger.info(
        "Validation Recall: %.4f",
        best_row["recall"]
    )

    logger.info(
        "Validation F1: %.4f",
        best_row["f1"]
    )

    return (
        best_threshold,
        threshold_df
    )


# ============================================================
# Calculate Final Test Metrics
# ============================================================

def calculate_test_metrics(
    y_test,
    y_prob,
    threshold
):

    # Apply threshold selected from validation
    y_pred = (
        y_prob >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    # ROC-AUC MUST use probabilities
    roc_auc = roc_auc_score(
        y_test,
        y_prob
    )

    metrics = {

        "threshold": float(
            threshold
        ),

        "accuracy": float(
            accuracy
        ),

        "precision": float(
            precision
        ),

        "recall": float(
            recall
        ),

        "f1_score": float(
            f1
        ),

        "roc_auc": float(
            roc_auc
        )
    }

    return metrics, y_pred


# ============================================================
# Save Metrics
# ============================================================

def save_metrics(metrics):

    metrics_path = os.path.join(
        REPORT_DIR,
        "metrics.json"
    )

    with open(
        metrics_path,
        "w"
    ) as f:

        json.dump(
            metrics,
            f,
            indent=4
        )

    logger.info(
        "Metrics saved to %s",
        metrics_path
    )


# ============================================================
# Save Predictions
# ============================================================

def save_predictions(
    y_test,
    y_prob,
    y_pred
):

    predictions = pd.DataFrame({

        "Actual": y_test,

        "Probability": y_prob,

        "Prediction": y_pred

    })

    predictions_path = os.path.join(
        REPORT_DIR,
        "predictions.csv"
    )

    predictions.to_csv(
        predictions_path,
        index=False
    )

    logger.info(
        "Predictions saved to %s",
        predictions_path
    )


# ============================================================
# Save Confusion Matrix
# ============================================================

def save_confusion_matrix(
    y_test,
    y_pred
):

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    logger.info(
        "Confusion Matrix:\n%s",
        cm
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "Stayed",
            "Churned"
        ]
    )

    disp.plot()

    plt.title(
        "XGBoost Customer Churn Confusion Matrix"
    )

    plt.tight_layout()

    confusion_path = os.path.join(
        REPORT_DIR,
        "confusion_matrix.png"
    )

    plt.savefig(
        confusion_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    logger.info(
        "Confusion matrix saved to %s",
        confusion_path
    )


# ============================================================
# Save Threshold
# ============================================================

def save_threshold(
    threshold
):

    threshold_path = os.path.join(
        REPORT_DIR,
        "threshold.txt"
    )

    with open(
        threshold_path,
        "w"
    ) as f:

        f.write(
            str(threshold)
        )

    logger.info(
        "Best threshold saved to %s",
        threshold_path
    )


# ============================================================
# Print Final Results
# ============================================================

def print_results(
    metrics,
    y_test,
    y_pred
):

    print("\n")
    print("=" * 60)
    print("FINAL XGBOOST TEST RESULTS")
    print("=" * 60)

    print(
        f"Threshold : {metrics['threshold']:.2f}"
    )

    print(
        f"Accuracy  : {metrics['accuracy']:.4f}"
    )

    print(
        f"Precision : {metrics['precision']:.4f}"
    )

    print(
        f"Recall    : {metrics['recall']:.4f}"
    )

    print(
        f"F1 Score  : {metrics['f1_score']:.4f}"
    )

    print(
        f"ROC-AUC   : {metrics['roc_auc']:.4f}"
    )

    print("=" * 60)

    print("\nClassification Report:")
    print("=" * 60)

    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )

    print("Confusion Matrix:")
    print(
        confusion_matrix(
            y_test,
            y_pred
        )
    )


# ============================================================
# Main
# ============================================================

def main():

    try:

        # Create reports directory
        os.makedirs(
            REPORT_DIR,
            exist_ok=True
        )

        # ----------------------------------------------------
        # 1. Load model
        # ----------------------------------------------------

        model = load_model()

        # ----------------------------------------------------
        # 2. Load validation data
        # ----------------------------------------------------

        x_val, y_val = (
            load_validation_data()
        )

        # ----------------------------------------------------
        # 3. Validation probabilities
        # ----------------------------------------------------

        logger.info(
            "Generating validation probabilities..."
        )

        val_prob = model.predict_proba(
            x_val
        )[:, 1]

        logger.info(
            "Validation probabilities generated."
        )

        # ----------------------------------------------------
        # 4. Find best threshold
        # ----------------------------------------------------

        (
            best_threshold,
            threshold_df
        ) = find_best_threshold(
            y_val,
            val_prob
        )

        # ----------------------------------------------------
        # 5. Save threshold experiments
        # ----------------------------------------------------

        threshold_results_path = os.path.join(
            REPORT_DIR,
            "threshold_results.csv"
        )

        threshold_df.to_csv(
            threshold_results_path,
            index=False
        )

        logger.info(
            "Threshold results saved to %s",
            threshold_results_path
        )

        # ----------------------------------------------------
        # 6. Save best threshold
        # ----------------------------------------------------

        save_threshold(
            best_threshold
        )

        # ----------------------------------------------------
        # 7. Load TEST data
        # ----------------------------------------------------

        x_test, y_test = (
            load_test_data()
        )

        # ----------------------------------------------------
        # 8. Generate TEST probabilities
        # ----------------------------------------------------

        logger.info(
            "Generating test probabilities..."
        )

        test_prob = model.predict_proba(
            x_test
        )[:, 1]

        # ----------------------------------------------------
        # 9. Final TEST evaluation
        # ----------------------------------------------------

        logger.info(
            "Evaluating final model on test data..."
        )

        metrics, y_pred = (
            calculate_test_metrics(
                y_test,
                test_prob,
                best_threshold
            )
        )

        # ----------------------------------------------------
        # 10. Print results
        # ----------------------------------------------------

        print_results(
            metrics,
            y_test,
            y_pred
        )

        # ----------------------------------------------------
        # 11. Save metrics
        # ----------------------------------------------------

        save_metrics(
            metrics
        )

        # ----------------------------------------------------
        # 12. Save predictions
        # ----------------------------------------------------

        save_predictions(
            y_test,
            test_prob,
            y_pred
        )

        # ----------------------------------------------------
        # 13. Save confusion matrix
        # ----------------------------------------------------

        save_confusion_matrix(
            y_test,
            y_pred
        )

        logger.info(
            "Evaluation pipeline completed successfully."
        )

    except FileNotFoundError as e:

        logger.error(
            "Required file not found: %s",
            e
        )

        raise

    except Exception as e:

        logger.exception(
            "Evaluation pipeline failed."
        )

        raise


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()