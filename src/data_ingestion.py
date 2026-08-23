import pandas as pd
import logging
import os

from sklearn.model_selection import train_test_split


# ============================================================
# Configuration
# ============================================================

DATA_FILE = "Churn_Modelling.csv"

DATA_DIR = "data"
RAW_DIR = os.path.join(DATA_DIR, "raw")

TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20

RANDOM_STATE = 42


# ============================================================
# Logging
# ============================================================

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("data_ingestion")
logger.setLevel(logging.DEBUG)

if not logger.handlers:

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    file_handler = logging.FileHandler(
        os.path.join(
            LOG_DIR,
            "data_ingestion.log"
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
# Load Data
# ============================================================

def load_data():

    logger.info(
        "Loading dataset from %s",
        DATA_FILE
    )

    df = pd.read_csv(DATA_FILE)

    logger.info(
        "Dataset shape: %s",
        df.shape
    )

    return df


# ============================================================
# Clean Data
# ============================================================

def clean_data(df):

    df = df.copy()

    columns_to_drop = [
        "RowNumber",
        "CustomerId",
        "Surname"
    ]

    df.drop(
        columns=columns_to_drop,
        inplace=True,
        errors="ignore"
    )

    logger.info(
        "Removed identifier columns"
    )

    return df


# ============================================================
# Save Data
# ============================================================

def save_data(
    x_train,
    x_val,
    x_test,
    y_train,
    y_val,
    y_test
):

    os.makedirs(
        RAW_DIR,
        exist_ok=True
    )

    x_train.to_csv(
        os.path.join(
            RAW_DIR,
            "x_train.csv"
        ),
        index=False
    )

    x_val.to_csv(
        os.path.join(
            RAW_DIR,
            "x_val.csv"
        ),
        index=False
    )

    x_test.to_csv(
        os.path.join(
            RAW_DIR,
            "x_test.csv"
        ),
        index=False
    )

    y_train.to_csv(
        os.path.join(
            RAW_DIR,
            "y_train.csv"
        ),
        index=False
    )

    y_val.to_csv(
        os.path.join(
            RAW_DIR,
            "y_val.csv"
        ),
        index=False
    )

    y_test.to_csv(
        os.path.join(
            RAW_DIR,
            "y_test.csv"
        ),
        index=False
    )

    logger.info(
        "Train/validation/test datasets saved."
    )


# ============================================================
# Main
# ============================================================

def main():

    try:

        df = load_data()

        df = clean_data(df)

        X = df.drop(
            columns=["Exited"]
        )

        y = df["Exited"]

        # ----------------------------------------------------
        # First: separate test set
        # ----------------------------------------------------

        X_temp, X_test, y_temp, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y
        )

        # ----------------------------------------------------
        # Second: separate validation set
        # ----------------------------------------------------

        X_train, X_val, y_train, y_val = train_test_split(
            X_temp,
            y_temp,
            test_size=VALIDATION_SIZE,
            random_state=RANDOM_STATE,
            stratify=y_temp
        )

        logger.info(
            "Train shape: %s",
            X_train.shape
        )

        logger.info(
            "Validation shape: %s",
            X_val.shape
        )

        logger.info(
            "Test shape: %s",
            X_test.shape
        )

        save_data(
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test
        )

        logger.info(
            "Data ingestion completed successfully."
        )

    except Exception as e:

        logger.exception(
            "Data ingestion failed: %s",
            e
        )

        raise


if __name__ == "__main__":
    main()