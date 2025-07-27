from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np
import pandas as pd
import logging
import os

# --- Logging Setup ---
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('preprocessing')
logger.setLevel(logging.DEBUG)

# Prevent duplicate handlers
if not logger.hasHandlers():
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(os.path.join(log_dir, 'preprocessing.log'))
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

# --- Preprocessing Function ---
def preprocess(x_train, x_test):
    # Step 1: One-Hot Encoding
    ohe = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)
    x_train_cat = ohe.fit_transform(x_train[["Geography", "Gender"]])
    x_test_cat = ohe.transform(x_test[["Geography", "Gender"]])

    # Step 2: Standard Scaling
    scaler = StandardScaler()
    x_train_num = x_train.drop(columns=["Geography", "Gender"]).values
    x_test_num = x_test.drop(columns=["Geography", "Gender"]).values

    x_train_scaled = scaler.fit_transform(x_train_num)
    x_test_scaled = scaler.transform(x_test_num)

    # Step 3: Combine Encoded + Scaled
    x_train_processed = np.hstack((x_train_scaled, x_train_cat))
    x_test_processed = np.hstack((x_test_scaled, x_test_cat))

    # Save to data/interim
    data_path = os.path.join("data", "interim")
    os.makedirs(data_path, exist_ok=True)

    pd.DataFrame(x_train_processed).to_csv(os.path.join(data_path, "x_train_processed.csv"), index=False)
    pd.DataFrame(x_test_processed).to_csv(os.path.join(data_path, "x_test_processed.csv"), index=False)

    logger.debug('Processed data saved to %s', data_path)

# --- Main Function ---
def main():
    """Load raw data, preprocess, and save processed outputs."""
    try:
        x_train = pd.read_csv('./data/raw/x_train.csv')
        x_test = pd.read_csv('./data/raw/x_test.csv')
        logger.debug('Raw data loaded successfully.')
        preprocess(x_train, x_test)
    except FileNotFoundError as e:
        logger.error('File not found: %s', e)
    except Exception as e:
        logger.error('Unexpected error: %s', e)

if __name__ == '__main__':
    main()
