import pandas as pd
import numpy as np
import logging
import os
import joblib
import shap
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = "data/output/xgb_model.pkl"

TEST_DATA_PATH = (
    "data/interim/x_test_processed.csv"
)

REPORT_DIR = "reports"
LOG_DIR = "logs"

TOP_FEATURES = [
    "Age",
    "NumOfProducts",
    "Gender_Male",
    "Geography_Germany",
    "ActiveAge"
]


# ============================================================
# Logging
# ============================================================

os.makedirs(
    LOG_DIR,
    exist_ok=True
)

logger = logging.getLogger("shap_explain")
logger.setLevel(logging.DEBUG)

if not logger.handlers:

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    file_handler = logging.FileHandler(
        os.path.join(
            LOG_DIR,
            "shap.log"
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

        os.makedirs(
            REPORT_DIR,
            exist_ok=True
        )

        logger.info(
            "Starting SHAP analysis..."
        )

        # ====================================================
        # 1. Load model
        # ====================================================

        logger.info(
            "Loading XGBoost model..."
        )

        if not os.path.exists(MODEL_PATH):

            raise FileNotFoundError(
                f"Model not found: {MODEL_PATH}"
            )

        model = joblib.load(
            MODEL_PATH
        )

        logger.info(
            "XGBoost model loaded successfully."
        )

        # ====================================================
        # 2. Load test data
        # ====================================================

        x_test = pd.read_csv(
            TEST_DATA_PATH
        )

        logger.info(
            "Test data shape: %s",
            x_test.shape
        )

        # ====================================================
        # 3. Create SHAP explainer
        # ====================================================

        logger.info(
            "Creating SHAP TreeExplainer..."
        )

        explainer = shap.TreeExplainer(
            model
        )

        # ====================================================
        # 4. Calculate SHAP values
        # ====================================================

        logger.info(
            "Calculating SHAP values..."
        )

        shap_values = explainer(
            x_test
        )

        logger.info(
            "SHAP values calculated successfully."
        )

        # ====================================================
        # 5. SHAP Beeswarm
        # ====================================================

        logger.info(
            "Creating SHAP summary plot..."
        )

        plt.figure(
            figsize=(10, 8)
        )

        shap.plots.beeswarm(
            shap_values,
            max_display=15,
            show=False
        )

        plt.tight_layout()

        summary_path = os.path.join(
            REPORT_DIR,
            "shap_summary.png"
        )

        plt.savefig(
            summary_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        logger.info(
            "SHAP summary saved to %s",
            summary_path
        )

        # ====================================================
        # 6. Mean Absolute SHAP Importance
        # ====================================================

        feature_importance = pd.DataFrame({

            "Feature": x_test.columns,

            "MeanAbsSHAP": np.abs(
                shap_values.values
            ).mean(axis=0)

        })

        feature_importance.sort_values(
            "MeanAbsSHAP",
            ascending=False,
            inplace=True
        )

        importance_path = os.path.join(
            REPORT_DIR,
            "shap_feature_importance.csv"
        )

        feature_importance.to_csv(
            importance_path,
            index=False
        )

        logger.info(
            "SHAP feature importance saved to %s",
            importance_path
        )

        # ====================================================
        # 7. SHAP Bar Plot
        # ====================================================

        logger.info(
            "Creating SHAP bar plot..."
        )

        plt.figure(
            figsize=(10, 8)
        )

        shap.plots.bar(
            shap_values,
            max_display=15,
            show=False
        )

        plt.tight_layout()

        bar_path = os.path.join(
            REPORT_DIR,
            "shap_feature_importance.png"
        )

        plt.savefig(
            bar_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        logger.info(
            "SHAP bar plot saved to %s",
            bar_path
        )

        # ====================================================
        # 8. Dependence Plots
        # ====================================================

        logger.info(
            "Creating SHAP dependence plots..."
        )

        for feature in TOP_FEATURES:

            # Check whether feature exists
            if feature not in x_test.columns:

                logger.warning(
                    "Feature '%s' not found. Skipping.",
                    feature
                )

                continue

            logger.info(
                "Creating dependence plot for %s",
                feature
            )

            plt.figure(
                figsize=(8, 6)
            )

            shap.dependence_plot(
                feature,
                shap_values.values,
                x_test,
                feature_names=x_test.columns,
                interaction_index="auto",
                show=False
            )

            plt.title(
                f"SHAP Dependence Plot - {feature}"
            )

            plt.tight_layout()

            # Make filename Linux-safe
            filename = (
                feature
                .replace("/", "_")
                .replace(" ", "_")
            )

            plot_path = os.path.join(
                REPORT_DIR,
                f"shap_dependence_{filename}.png"
            )

            plt.savefig(
                plot_path,
                dpi=300,
                bbox_inches="tight"
            )

            plt.close()

            logger.info(
                "Saved: %s",
                plot_path
            )

        # ====================================================
        # 9. Print Top Features
        # ====================================================

        print("\n")
        print("=" * 60)
        print("TOP SHAP FEATURES")
        print("=" * 60)

        print(
            feature_importance.head(15).to_string(
                index=False
            )
        )

        print("=" * 60)

        logger.info(
            "SHAP pipeline completed successfully."
        )

    except FileNotFoundError as e:

        logger.error(
            "Required file not found: %s",
            e
        )

        raise

    except Exception as e:

        logger.exception(
            "SHAP analysis failed."
        )

        raise


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()