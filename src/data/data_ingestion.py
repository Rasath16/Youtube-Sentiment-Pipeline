import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import yaml
import logging

# Logging configuration
logger = logging.getLogger('data_ingestion')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('errors.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise

def load_data(data_url: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(data_url)
        logger.debug('Data loaded from %s with shape %s', data_url, df.shape)
        logger.debug('Columns: %s', df.columns.tolist())
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data by handling missing values, duplicates, and empty strings."""
    try:
        initial_shape = df.shape
        logger.debug('Initial data shape: %s', initial_shape)
        
        # Removing missing values
        df.dropna(inplace=True)
        logger.debug('After removing missing values: %s rows', df.shape[0])
        
        # Removing duplicates
        df.drop_duplicates(inplace=True)
        logger.debug('After removing duplicates: %s rows', df.shape[0])
        
        # Removing rows with empty strings
        df = df[df['clean_comment'].str.strip() != '']
        logger.debug('After removing empty strings: %s rows', df.shape[0])
        
        # Log class distribution
        class_distribution = df['category'].value_counts().to_dict()
        logger.debug('Class distribution: %s', class_distribution)
        
        logger.debug('Data preprocessing completed: %s rows remaining from %s', 
                    df.shape[0], initial_shape[0])
        return df
    except KeyError as e:
        logger.error('Missing column in the dataframe: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error during preprocessing: %s', e)
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save the train and test datasets, creating the raw folder if it doesn't exist."""
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        
        # Create the data/raw directory if it does not exist
        os.makedirs(raw_data_path, exist_ok=True)
        
        # Save the train and test data
        train_path = os.path.join(raw_data_path, "train.csv")
        test_path = os.path.join(raw_data_path, "test.csv")
        
        train_data.to_csv(train_path, index=False)
        test_data.to_csv(test_path, index=False)
        
        logger.debug('Train data saved: %s with shape %s', train_path, train_data.shape)
        logger.debug('Test data saved: %s with shape %s', test_path, test_data.shape)
        
        # Log train/test class distributions
        logger.debug('Train class distribution: %s', train_data['category'].value_counts().to_dict())
        logger.debug('Test class distribution: %s', test_data['category'].value_counts().to_dict())
        
    except Exception as e:
        logger.error('Unexpected error occurred while saving the data: %s', e)
        raise

def main():
    try:
        logger.info('='*80)
        logger.info('Starting Data Ingestion Process')
        logger.info('='*80)
        
        # Load parameters from the params.yaml in the root directory
        params_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../params.yaml')
        params = load_params(params_path=params_path)
        
        # Extract parameters
        test_size = params['data_ingestion']['test_size']
        random_state = params['data_ingestion']['random_state']
        stratify = params['data_ingestion']['stratify']
        
        logger.info('Parameters loaded:')
        logger.info('  - test_size: %s', test_size)
        logger.info('  - random_state: %s', random_state)
        logger.info('  - stratify: %s', stratify)
        
        # Load data from the specified URL
        data_url = 'https://raw.githubusercontent.com/Himanshu-1703/reddit-sentiment-analysis/refs/heads/main/data/reddit.csv'
        logger.info('Loading data from: %s', data_url)
        df = load_data(data_url=data_url)
        
        # Preprocess the data
        logger.info('Preprocessing data...')
        final_df = preprocess_data(df)
        
        # Split the data into training and testing sets
        logger.info('Splitting data into train and test sets...')
        if stratify:
            train_data, test_data = train_test_split(
                final_df, 
                test_size=test_size, 
                random_state=random_state,
                stratify=final_df['category']
            )
            logger.info('Stratified split completed')
        else:
            train_data, test_data = train_test_split(
                final_df, 
                test_size=test_size, 
                random_state=random_state
            )
            logger.info('Random split completed')
        
        # Save the split datasets
        data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data')
        logger.info('Saving train and test data...')
        save_data(train_data, test_data, data_path=data_path)
        
        logger.info('='*80)
        logger.info('Data Ingestion Process Completed Successfully!')
        logger.info('='*80)
        
    except Exception as e:
        logger.error('='*80)
        logger.error('Failed to complete the data ingestion process: %s', e)
        logger.error('='*80)
        print(f"Error: {e}")
        raise

if __name__ == '__main__':
    main()