from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os


app = Flask(__name__)


# ============================================================
# Paths
# ============================================================

MODEL_PATH = "data/output/xgb_model.pkl"

ENCODER_PATH = "data/interim/encoder.pkl"

FEATURE_NAMES_PATH = (
    "data/interim/feature_names.txt"
)

THRESHOLD_PATH = (
    "reports/threshold.txt"
)


# ============================================================
# Load Model and Preprocessing Artifacts
# ============================================================

print("Loading model and preprocessing artifacts...")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

if not os.path.exists(ENCODER_PATH):
    raise FileNotFoundError(
        f"Encoder not found: {ENCODER_PATH}"
    )

if not os.path.exists(FEATURE_NAMES_PATH):
    raise FileNotFoundError(
        f"Feature names not found: {FEATURE_NAMES_PATH}"
    )

if not os.path.exists(THRESHOLD_PATH):
    raise FileNotFoundError(
        f"Threshold not found: {THRESHOLD_PATH}"
    )


model = joblib.load(
    MODEL_PATH
)

encoder = joblib.load(
    ENCODER_PATH
)


# Load feature names
with open(
    FEATURE_NAMES_PATH,
    "r"
) as f:

    feature_names = [
        line.strip()
        for line in f
        if line.strip()
    ]


# Load validation-selected threshold
with open(
    THRESHOLD_PATH,
    "r"
) as f:

    threshold = float(
        f.read().strip()
    )


print(
    "XGBoost model loaded successfully."
)

print(
    f"Classification threshold: {threshold}"
)

print(
    f"Number of features: {len(feature_names)}"
)


# ============================================================
# Feature Engineering
# ============================================================

def create_features(df):

    df = df.copy()

    # --------------------------------------------------------
    # Engineered features
    # --------------------------------------------------------

    df["ZeroBalance"] = (
        df["Balance"] == 0
    ).astype(int)

    df["BalanceSalaryRatio"] = (
        df["Balance"] /
        (df["EstimatedSalary"] + 1)
    )

    df["BalancePerProduct"] = (
        df["Balance"] /
        (df["NumOfProducts"] + 1)
    )

    df["SalaryPerProduct"] = (
        df["EstimatedSalary"] /
        (df["NumOfProducts"] + 1)
    )

    df["CreditAgeInteraction"] = (
        df["CreditScore"] *
        df["Age"]
    )

    df["ActiveTenure"] = (
        df["IsActiveMember"] *
        df["Tenure"]
    )

    df["ActiveAge"] = (
        df["IsActiveMember"] *
        df["Age"]
    )

    df["ProductsActive"] = (
        df["NumOfProducts"] *
        df["IsActiveMember"]
    )

    df["TenureAgeRatio"] = (
        df["Tenure"] /
        (df["Age"] + 1)
    )

    return df


# ============================================================
# Preprocess Input
# ============================================================

def preprocess_input(df):

    df = create_features(df)

    categorical_columns = [
        "Geography",
        "Gender"
    ]

    numerical_columns = [
        col
        for col in df.columns
        if col not in categorical_columns
    ]

    # --------------------------------------------------------
    # Numerical features
    # --------------------------------------------------------

    numerical_data = df[
        numerical_columns
    ].reset_index(drop=True)

    # --------------------------------------------------------
    # Categorical features
    # Use SAME encoder fitted during training
    # --------------------------------------------------------

    categorical_data = encoder.transform(
        df[categorical_columns]
    )

    categorical_data = pd.DataFrame(
        categorical_data,
        columns=encoder.get_feature_names_out(
            categorical_columns
        )
    )

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    processed = pd.concat(
        [
            numerical_data,
            categorical_data
        ],
        axis=1
    )

    # --------------------------------------------------------
    # Handle infinity / NaN
    # --------------------------------------------------------

    processed.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )

    processed.fillna(
        0,
        inplace=True
    )

    # --------------------------------------------------------
    # EXACT training feature order
    # --------------------------------------------------------

    processed = processed.reindex(
        columns=feature_names,
        fill_value=0
    )

    return processed


# ============================================================
# Home / Health Check
# ============================================================

@app.route(
    "/",
    methods=["GET"]
)
def home():

    return jsonify({

        "message":
            "Customer Churn Prediction API",

        "status":
            "running",

        "model":
            "XGBoost",

        "threshold":
            threshold

    })


# ============================================================
# Prediction API
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        # ----------------------------------------------------
        # Get JSON
        # ----------------------------------------------------

        data = request.get_json()

        if data is None:

            return jsonify({
                "error":
                    "Request body must contain JSON data."
            }), 400

        # ----------------------------------------------------
        # Convert to DataFrame
        # ----------------------------------------------------

        df = pd.DataFrame(
            [data]
        )

        # ----------------------------------------------------
        # Validate required columns
        # ----------------------------------------------------

        required_columns = [

            "CreditScore",
            "Geography",
            "Gender",
            "Age",
            "Tenure",
            "Balance",
            "NumOfProducts",
            "HasCrCard",
            "IsActiveMember",
            "EstimatedSalary"

        ]

        missing_columns = [
            col
            for col in required_columns
            if col not in df.columns
        ]

        if missing_columns:

            return jsonify({

                "error":
                    "Missing required features.",

                "missing_features":
                    missing_columns

            }), 400

        # ----------------------------------------------------
        # Preprocess
        # ----------------------------------------------------

        processed_data = preprocess_input(
            df
        )

        # ----------------------------------------------------
        # Predict probability
        # ----------------------------------------------------

        probability = model.predict_proba(
            processed_data
        )[0][1]

        # ----------------------------------------------------
        # Apply validation-selected threshold
        # ----------------------------------------------------

        prediction = int(
            probability >= threshold
        )

        # ----------------------------------------------------
        # Response
        # ----------------------------------------------------

        return jsonify({

            "churn_probability":
                round(
                    float(probability),
                    4
                ),

            "prediction":
                prediction,

            "churn":
                (
                    "Yes"
                    if prediction == 1
                    else "No"
                ),

            "threshold":
                threshold

        })

    except Exception as e:

        return jsonify({

            "error":
                str(e)

        }), 500


# ============================================================
# Run Flask
# ============================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000
    )