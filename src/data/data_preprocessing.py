import numpy as np
import pandas as pd
import os
import re
import nltk
import yaml
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import logging

# Logging configuration
logger = logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('preprocessing_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Download required NLTK data
try:
    nltk.data.find('corpora/wordnet')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('wordnet', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('omw-1.4', quiet=True)  # Required for wordnet

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

def preprocess_comment(comment: str, 
                      remove_stopwords: bool = True,
                      lowercase: bool = True,
                      remove_punctuation: bool = False,
                      remove_numbers: bool = False,
                      lemmatization: bool = True) -> str:
    """
    Apply preprocessing transformations to a comment based on parameters.
    
    Args:
        comment: Text to preprocess
        remove_stopwords: Whether to remove stopwords
        lowercase: Whether to convert to lowercase
        remove_punctuation: Whether to remove punctuation
        remove_numbers: Whether to remove numbers
        lemmatization: Whether to apply lemmatization
    
    Returns:
        Preprocessed text
    """
    try:
        # Convert to lowercase
        if lowercase:
            comment = comment.lower()
        
        # Remove trailing and leading whitespaces
        comment = comment.strip()
        
        # Remove newline characters
        comment = re.sub(r'\n', ' ', comment)
        
        # Remove multiple spaces
        comment = re.sub(r'\s+', ' ', comment)
        
        # Remove numbers if specified
        if remove_numbers:
            comment = re.sub(r'\d+', '', comment)
        
        # Remove punctuation if specified, otherwise keep some for context
        if remove_punctuation:
            comment = comment.translate(str.maketrans('', '', string.punctuation))
        else:
            # Keep important punctuation, remove others
            comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)
        
        # Remove stopwords but retain important ones for sentiment analysis
        if remove_stopwords:
            stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet', 'nor', 'neither'}
            comment = ' '.join([word for word in comment.split() if word not in stop_words])
        
        # Lemmatize the words
        if lemmatization:
            lemmatizer = WordNetLemmatizer()
            comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])
        
        # Final cleanup - remove extra spaces
        comment = ' '.join(comment.split())
        
        return comment
    except Exception as e:
        logger.error(f"Error in preprocessing comment: {e}")
        return comment

def normalize_text(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Apply preprocessing to the text data in the dataframe.
    
    Args:
        df: DataFrame with 'clean_comment' column
        params: Dictionary with preprocessing parameters
    
    Returns:
        DataFrame with preprocessed text
    """
    try:
        initial_count = len(df)
        logger.debug(f'Starting text normalization on {initial_count} samples')
        
        # CRITICAL: Remap labels to [0, 1, 2] for model compatibility
        # Original: -1 (negative), 0 (neutral), 1 (positive)
        # Remapped: 2 (negative), 0 (neutral), 1 (positive)
        if 'category' in df.columns:
            logger.info('Remapping category labels: -1→2, 0→0, 1→1')
            df['category'] = df['category'].map({-1: 2, 0: 0, 1: 1})
            df = df.dropna(subset=['category'])  # Remove any unmapped values
            logger.info(f'Class distribution after remapping: {df["category"].value_counts().sort_index().to_dict()}')
        
        # Extract preprocessing parameters
        preproc_params = params['text_preprocessing']
        
        # Apply preprocessing with parameters
        df['clean_comment'] = df['clean_comment'].apply(
            lambda x: preprocess_comment(
                x,
                remove_stopwords=preproc_params['remove_stopwords'],
                lowercase=preproc_params['lowercase'],
                remove_punctuation=preproc_params['remove_punctuation'],
                remove_numbers=preproc_params['remove_numbers'],
                lemmatization=preproc_params['lemmatization']
            )
        )
        
        # Remove any rows that became empty after preprocessing
        df = df[df['clean_comment'].str.strip() != '']
        final_count = len(df)
        
        if final_count < initial_count:
            logger.warning(f'Removed {initial_count - final_count} empty rows after preprocessing')
        
        logger.debug(f'Text normalization completed on {final_count} samples')
        
        # Log sample of processed data
        logger.debug('Sample processed comments:')
        for idx, comment in df['clean_comment'].head(3).items():
            logger.debug(f'  Sample {idx}: {comment[:100]}...')
        
        return df
    except Exception as e:
        logger.error(f"Error during text normalization: {e}")
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save the processed train and test datasets."""
    try:
        interim_data_path = os.path.join(data_path, 'interim')
        logger.debug(f"Creating directory {interim_data_path}")
        
        # Ensure the directory is created
        os.makedirs(interim_data_path, exist_ok=True)
        logger.debug(f"Directory {interim_data_path} created or already exists")
        
        # Save files
        train_path = os.path.join(interim_data_path, "train_processed.csv")
        test_path = os.path.join(interim_data_path, "test_processed.csv")
        
        train_data.to_csv(train_path, index=False)
        test_data.to_csv(test_path, index=False)
        
        logger.debug(f"Train data saved: {train_path} with shape {train_data.shape}")
        logger.debug(f"Test data saved: {test_path} with shape {test_data.shape}")
        
        # Log class distributions
        logger.debug(f"Train class distribution: {train_data['category'].value_counts().to_dict()}")
        logger.debug(f"Test class distribution: {test_data['category'].value_counts().to_dict()}")
        
    except Exception as e:
        logger.error(f"Error occurred while saving data: {e}")
        raise

def main():
    try:
        logger.info('='*80)
        logger.info('Starting Data Preprocessing Process')
        logger.info('='*80)
        
        # Load parameters
        params_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../params.yaml')
        params = load_params(params_path)
        
        logger.info('Preprocessing parameters:')
        for key, value in params['text_preprocessing'].items():
            logger.info(f'  - {key}: {value}')
        
        # Load data from data/raw
        train_path = './data/raw/train.csv'
        test_path = './data/raw/test.csv'
        
        logger.info(f'Loading train data from: {train_path}')
        train_data = pd.read_csv(train_path)
        logger.info(f'Train data loaded: {train_data.shape}')
        
        logger.info(f'Loading test data from: {test_path}')
        test_data = pd.read_csv(test_path)
        logger.info(f'Test data loaded: {test_data.shape}')
        
        # Preprocess the data
        logger.info('Preprocessing training data...')
        train_processed_data = normalize_text(train_data, params)
        
        logger.info('Preprocessing test data...')
        test_processed_data = normalize_text(test_data, params)
        
        # Save the processed data
        logger.info('Saving processed data...')
        save_data(train_processed_data, test_processed_data, data_path='./data')
        
        logger.info('='*80)
        logger.info('Data Preprocessing Process Completed Successfully!')
        logger.info(f'Final train samples: {len(train_processed_data)}')
        logger.info(f'Final test samples: {len(test_processed_data)}')
        logger.info('='*80)
        
    except Exception as e:
        logger.error('='*80)
        logger.error('Failed to complete the data preprocessing process: %s', e)
        logger.error('='*80)
        print(f"Error: {e}")
        raise

if __name__ == '__main__':
    main()