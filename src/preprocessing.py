import pandas as pd
import numpy as np
import logging
import os
import joblib

from sklearn.preprocessing import OneHotEncoder


# ============================================================
# Configuration
# ============================================================

RAW_DIR = "data/raw"
INTERIM_DIR = "data/interim"
LOG_DIR = "logs"

CATEGORICAL_COLUMNS = [
    "Geography",
    "Gender"
]


# ============================================================
# Logging
# ============================================================

os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("preprocessing")
logger.setLevel(logging.DEBUG)

if not logger.handlers:

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    file_handler = logging.FileHandler(
        os.path.join(
            LOG_DIR,
            "preprocessing.log"
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
# Feature Engineering
# ============================================================

def create_features(df):

    df = df.copy()

    # 1. Zero balance
    df["ZeroBalance"] = (
        df["Balance"] == 0
    ).astype(int)

    # 2. Balance / salary
    df["BalanceSalaryRatio"] = (
        df["Balance"] /
        (df["EstimatedSalary"] + 1)
    )

    # 3. Balance per product
    df["BalancePerProduct"] = (
        df["Balance"] /
        (df["NumOfProducts"] + 1)
    )

    # 4. Salary per product
    df["SalaryPerProduct"] = (
        df["EstimatedSalary"] /
        (df["NumOfProducts"] + 1)
    )

    # 5. Credit score × age
    df["CreditAgeInteraction"] = (
        df["CreditScore"] *
        df["Age"]
    )

    # 6. Active × tenure
    df["ActiveTenure"] = (
        df["IsActiveMember"] *
        df["Tenure"]
    )

    # 7. Active × age
    df["ActiveAge"] = (
        df["IsActiveMember"] *
        df["Age"]
    )

    # 8. Products × active
    df["ProductsActive"] = (
        df["NumOfProducts"] *
        df["IsActiveMember"]
    )

    # 9. Tenure / age
    df["TenureAgeRatio"] = (
        df["Tenure"] /
        (df["Age"] + 1)
    )

    return df


# ============================================================
# Preprocessing
# ============================================================

def preprocess():

    logger.info(
        "Loading raw datasets..."
    )

    x_train = pd.read_csv(
        os.path.join(
            RAW_DIR,
            "x_train.csv"
        )
    )

    x_val = pd.read_csv(
        os.path.join(
            RAW_DIR,
            "x_val.csv"
        )
    )

    x_test = pd.read_csv(
        os.path.join(
            RAW_DIR,
            "x_test.csv"
        )
    )

    # --------------------------------------------------------
    # Feature engineering
    # --------------------------------------------------------

    x_train = create_features(x_train)
    x_val = create_features(x_val)
    x_test = create_features(x_test)

    logger.info(
        "Feature engineering completed."
    )

    # --------------------------------------------------------
    # Numerical columns
    # --------------------------------------------------------

    numerical_columns = [
        col
        for col in x_train.columns
        if col not in CATEGORICAL_COLUMNS
    ]

    # --------------------------------------------------------
    # Fit encoder ONLY on training data
    # --------------------------------------------------------

    ohe = OneHotEncoder(
        drop="first",
        handle_unknown="ignore",
        sparse_output=False
    )

    train_cat = ohe.fit_transform(
        x_train[CATEGORICAL_COLUMNS]
    )

    val_cat = ohe.transform(
        x_val[CATEGORICAL_COLUMNS]
    )

    test_cat = ohe.transform(
        x_test[CATEGORICAL_COLUMNS]
    )

    encoded_names = (
        ohe.get_feature_names_out(
            CATEGORICAL_COLUMNS
        )
    )

    # --------------------------------------------------------
    # Numerical data
    # --------------------------------------------------------

    train_num = x_train[
        numerical_columns
    ].reset_index(drop=True)

    val_num = x_val[
        numerical_columns
    ].reset_index(drop=True)

    test_num = x_test[
        numerical_columns
    ].reset_index(drop=True)

    # --------------------------------------------------------
    # Convert categorical arrays to DataFrames
    # --------------------------------------------------------

    train_cat = pd.DataFrame(
        train_cat,
        columns=encoded_names
    )

    val_cat = pd.DataFrame(
        val_cat,
        columns=encoded_names
    )

    test_cat = pd.DataFrame(
        test_cat,
        columns=encoded_names
    )

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    x_train_processed = pd.concat(
        [
            train_num,
            train_cat
        ],
        axis=1
    )

    x_val_processed = pd.concat(
        [
            val_num,
            val_cat
        ],
        axis=1
    )

    x_test_processed = pd.concat(
        [
            test_num,
            test_cat
        ],
        axis=1
    )

    # --------------------------------------------------------
    # Handle inf / NaN
    # --------------------------------------------------------

    x_train_processed.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )

    x_val_processed.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )

    x_test_processed.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )

    train_medians = (
        x_train_processed.median(
            numeric_only=True
        )
    )

    x_train_processed.fillna(
        train_medians,
        inplace=True
    )

    x_val_processed.fillna(
        train_medians,
        inplace=True
    )

    x_test_processed.fillna(
        train_medians,
        inplace=True
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    os.makedirs(
        INTERIM_DIR,
        exist_ok=True
    )

    x_train_processed.to_csv(
        os.path.join(
            INTERIM_DIR,
            "x_train_processed.csv"
        ),
        index=False
    )

    x_val_processed.to_csv(
        os.path.join(
            INTERIM_DIR,
            "x_val_processed.csv"
        ),
        index=False
    )

    x_test_processed.to_csv(
        os.path.join(
            INTERIM_DIR,
            "x_test_processed.csv"
        ),
        index=False
    )

    # Save encoder
    joblib.dump(
        ohe,
        os.path.join(
            INTERIM_DIR,
            "encoder.pkl"
        )
    )

    # Save feature names
    with open(
        os.path.join(
            INTERIM_DIR,
            "feature_names.txt"
        ),
        "w"
    ) as f:

        for feature in x_train_processed.columns:
            f.write(
                feature + "\n"
            )

    logger.info(
        "Preprocessing completed successfully."
    )

    logger.info(
        "Final feature count: %d",
        x_train_processed.shape[1]
    )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    preprocess()