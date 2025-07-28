import pandas as pd
import numpy as np
import logging
import yaml
import os
from sklearn.model_selection import train_test_split

# Ensure the "logs" directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)


# logging configuration
logger = logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'data_ingestion.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data():
    """Load data from a CSV file."""
    try:
        df = pd.read_csv("Churn_Modelling.csv")
        logger.debug('Data loaded from file')
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise

def save_data(x_train: pd.DataFrame, x_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame,data_path: str) -> None:
    """Save the train and test datasets."""
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        x_train.to_csv(os.path.join(raw_data_path, "x_train.csv"), index=False)
        x_test.to_csv(os.path.join(raw_data_path, "x_test.csv"), index=False)
        y_train.to_csv(os.path.join(raw_data_path, "y_train.csv"), index=False)
        y_test.to_csv(os.path.join(raw_data_path, "y_test.csv"), index=False)
        logger.debug('Train and test data saved to %s', raw_data_path)
    except Exception as e:
        logger.error('Unexpected error occurred while saving the data: %s', e)
        raise


def main():
    try:
        test_size = 0.2
        df=load_data()
        df.drop(columns=["RowNumber","Surname","CustomerId"],inplace=True)
        x_train,x_test,y_train,y_test=train_test_split(df.drop(columns=["Exited"]),df["Exited"],test_size=test_size,random_state=2)
        save_data(x_train,x_test,y_train,y_test, data_path='./data')
    except Exception as e:
        logger.error('Failed to complete the data ingestion process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()