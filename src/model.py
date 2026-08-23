import pandas as pd
import logging
import os
import json
import joblib

from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV


# ============================================================
# Configuration
# ============================================================

INTERIM_DIR = "data/interim"
RAW_DIR = "data/raw"
OUTPUT_DIR = "data/output"
LOG_DIR = "logs"

RANDOM_STATE = 42


# ============================================================
# Logging
# ============================================================

os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("model")
logger.setLevel(logging.DEBUG)

if not logger.handlers:

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    file_handler = logging.FileHandler(
        os.path.join(
            LOG_DIR,
            "model.log"
        )
    )

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - "
        "%(levelname)s - %(message)s"
    )

    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


# ============================================================
# Main
# ============================================================

def main():

    try:

        logger.info(
            "Loading training data..."
        )

        x_train = pd.read_csv(
            os.path.join(
                INTERIM_DIR,
                "x_train_processed.csv"
            )
        )

        y_train = pd.read_csv(
            os.path.join(
                RAW_DIR,
                "y_train.csv"
            )
        ).values.ravel()

        logger.info(
            "Training shape: %s",
            x_train.shape
        )

        # ----------------------------------------------------
        # XGBoost
        # ----------------------------------------------------

        xgb = XGBClassifier(
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            n_jobs=-1
        )

        param_distributions = {

            "n_estimators": [
                200,
                300,
                500
            ],

            "max_depth": [
                3,
                4,
                5,
                6
            ],

            "learning_rate": [
                0.01,
                0.03,
                0.05,
                0.1
            ],

            "subsample": [
                0.8,
                1.0
            ],

            "colsample_bytree": [
                0.8,
                1.0
            ],

            "min_child_weight": [
                1,
                3,
                5
            ],

            "gamma": [
                0,
                0.1,
                0.3
            ],

            "scale_pos_weight": [
                1,
                2,
                3
            ]
        }

        search = RandomizedSearchCV(
            estimator=xgb,
            param_distributions=param_distributions,
            n_iter=40,
            scoring="roc_auc",
            cv=5,
            random_state=RANDOM_STATE,
            verbose=1,
            n_jobs=-1
        )

        search.fit(
            x_train,
            y_train
        )

        best_model = search.best_estimator_

        logger.info(
            "Best parameters: %s",
            search.best_params_
        )

        logger.info(
            "Best CV ROC-AUC: %.4f",
            search.best_score_
        )

        # ----------------------------------------------------
        # Save model
        # ----------------------------------------------------

        os.makedirs(
            OUTPUT_DIR,
            exist_ok=True
        )

        joblib.dump(
            best_model,
            os.path.join(
                OUTPUT_DIR,
                "xgb_model.pkl"
            )
        )

        metadata = {
            "model": "XGBoost",
            "best_params": search.best_params_,
            "best_cv_roc_auc": float(
                search.best_score_
            )
        }

        with open(
            os.path.join(
                OUTPUT_DIR,
                "model_metadata.json"
            ),
            "w"
        ) as f:

            json.dump(
                metadata,
                f,
                indent=4
            )

        logger.info(
            "XGBoost model saved successfully."
        )

    except Exception as e:

        logger.exception(
            "Model training failed: %s",
            e
        )

        raise


if __name__ == "__main__":
    main()