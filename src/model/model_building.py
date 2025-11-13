import numpy as np
import pandas as pd
import os
import pickle
import yaml
import logging
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.under_sampling import RandomUnderSampler

# Logging configuration
logger = logging.getLogger('model_building')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_building_errors.log')
file_handler.setLevel('ERROR')

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


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        df.fillna('', inplace=True)  # Fill any NaN values
        logger.debug('Data loaded from %s with shape %s', file_path, df.shape)
        logger.debug('Class distribution: %s', df['category'].value_counts().to_dict())
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise


def apply_tfidf(train_data: pd.DataFrame, params: dict) -> tuple:
    """
    Apply TF-IDF vectorization to the training data.
    
    Args:
        train_data: DataFrame with 'clean_comment' and 'category' columns
        params: Dictionary with feature engineering parameters
    
    Returns:
        Tuple of (X_train_tfidf, y_train, vectorizer)
    """
    try:
        # Extract feature engineering parameters
        feat_params = params['feature_engineering']
        max_features = feat_params['max_features']
        ngram_range = tuple(feat_params['ngram_range'])
        min_df = feat_params['min_df']
        max_df = feat_params['max_df']
        use_idf = feat_params['use_idf']
        sublinear_tf = feat_params['sublinear_tf']
        
        logger.info('TF-IDF Configuration:')
        logger.info(f'  - max_features: {max_features}')
        logger.info(f'  - ngram_range: {ngram_range}')
        logger.info(f'  - min_df: {min_df}')
        logger.info(f'  - max_df: {max_df}')
        logger.info(f'  - use_idf: {use_idf}')
        logger.info(f'  - sublinear_tf: {sublinear_tf}')
        
        # Initialize TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            use_idf=use_idf,
            sublinear_tf=sublinear_tf
        )
        
        X_train = train_data['clean_comment'].values
        y_train = train_data['category'].values
        
        # Perform TF-IDF transformation
        logger.info('Fitting TF-IDF vectorizer on training data...')
        X_train_tfidf = vectorizer.fit_transform(X_train)
        
        logger.info(f'TF-IDF transformation complete')
        logger.info(f'  - Training shape: {X_train_tfidf.shape}')
        logger.info(f'  - Vocabulary size: {len(vectorizer.vocabulary_)}')
        logger.info(f'  - Feature names (sample): {vectorizer.get_feature_names_out()[:10]}')
        
        return X_train_tfidf, y_train, vectorizer
        
    except Exception as e:
        logger.error('Error during TF-IDF transformation: %s', e)
        raise


def handle_imbalance(X_train: np.ndarray, y_train: np.ndarray, params: dict) -> tuple:
    """
    Apply undersampling to handle class imbalance.
    
    Args:
        X_train: Training features
        y_train: Training labels
        params: Dictionary with imbalance handling parameters
    
    Returns:
        Tuple of (X_train_resampled, y_train_resampled)
    """
    try:
        # Extract imbalance handling parameters
        imb_params = params['imbalance_handling']
        method = imb_params['method']
        random_state = imb_params['random_state']
        sampling_strategy = imb_params['sampling_strategy']
        
        logger.info('Handling class imbalance:')
        logger.info(f'  - Method: {method}')
        logger.info(f'  - Sampling strategy: {sampling_strategy}')
        
        # Log original class distribution
        unique, counts = np.unique(y_train, return_counts=True)
        logger.info(f'Original class distribution: {dict(zip(unique, counts))}')
        
        if method == 'undersampling':
            sampler = RandomUnderSampler(
                random_state=random_state,
                sampling_strategy=sampling_strategy
            )
            X_train_resampled, y_train_resampled = sampler.fit_resample(X_train, y_train)
            
            # Log new class distribution
            unique, counts = np.unique(y_train_resampled, return_counts=True)
            logger.info(f'Resampled class distribution: {dict(zip(unique, counts))}')
            logger.info(f'Resampled data shape: {X_train_resampled.shape}')
            
            return X_train_resampled, y_train_resampled
        else:
            logger.warning(f'Unknown imbalance handling method: {method}. Skipping resampling.')
            return X_train, y_train
            
    except Exception as e:
        logger.error('Error during imbalance handling: %s', e)
        raise


def train_linearsvc(X_train: np.ndarray, y_train: np.ndarray, params: dict) -> LinearSVC:
    """
    Train a LinearSVC model.
    
    Args:
        X_train: Training features (after resampling)
        y_train: Training labels (after resampling)
        params: Dictionary with model building parameters
    
    Returns:
        Trained LinearSVC model
    """
    try:
        # Extract model building parameters
        model_params = params['model_building']
        
        logger.info('LinearSVC Configuration:')
        logger.info(f'  - C: {model_params["C"]}')
        logger.info(f'  - loss: {model_params["loss"]}')
        logger.info(f'  - penalty: {model_params["penalty"]}')
        logger.info(f'  - dual: {model_params["dual"]}')
        logger.info(f'  - max_iter: {model_params["max_iter"]}')
        logger.info(f'  - tol: {model_params["tol"]}')
        logger.info(f'  - class_weight: {model_params["class_weight"]}')
        
        # Initialize LinearSVC model
        model = LinearSVC(
            C=model_params['C'],
            loss=model_params['loss'],
            penalty=model_params['penalty'],
            dual=model_params['dual'],
            max_iter=model_params['max_iter'],
            tol=model_params['tol'],
            class_weight=model_params['class_weight'],
            random_state=model_params['random_state'],
            fit_intercept=model_params['fit_intercept'],
            intercept_scaling=model_params['intercept_scaling'],
            multi_class=model_params['multi_class'],
            verbose=model_params['verbose']
        )
        
        # Train the model
        logger.info('Training LinearSVC model...')
        model.fit(X_train, y_train)
        logger.info('LinearSVC model training completed successfully')
        
        # Log model information
        logger.info(f'Number of classes: {len(model.classes_)}')
        logger.info(f'Classes: {model.classes_}')
        logger.info(f'Number of iterations: {model.n_iter_}')
        
        return model
        
    except Exception as e:
        logger.error('Error during LinearSVC model training: %s', e)
        raise


def save_artifacts(vectorizer, model, root_dir: str) -> None:
    """
    Save the vectorizer and model to the models directory.
    
    Args:
        vectorizer: Trained TF-IDF vectorizer
        model: Trained LinearSVC model
        root_dir: Root directory path
    """
    try:
        # Create models directory if it doesn't exist
        models_dir = os.path.join(root_dir, 'models')
        os.makedirs(models_dir, exist_ok=True)
        
        # Save TF-IDF vectorizer
        vectorizer_path = os.path.join(models_dir, 'tfidf_vectorizer.pkl')
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(vectorizer, f)
        logger.info(f'TF-IDF vectorizer saved to {vectorizer_path}')
        
        # Save LinearSVC model
        model_path = os.path.join(models_dir, 'linearsvc_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f'LinearSVC model saved to {model_path}')
        
        # Save resampled data for evaluation
        # Note: In a production pipeline, you might want to save these as well
        # but for now we'll regenerate them in evaluation
        
    except Exception as e:
        logger.error('Error occurred while saving artifacts: %s', e)
        raise


def get_root_directory() -> str:
    """Get the root directory (two levels up from this script's location)."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, '../../'))


def main():
    try:
        logger.info('='*80)
        logger.info('Starting Model Building Process (LinearSVC)')
        logger.info('='*80)
        
        # Get root directory
        root_dir = get_root_directory()
        
        # Load parameters
        params_path = os.path.join(root_dir, 'params.yaml')
        params = load_params(params_path)
        
        # Load the preprocessed training data
        train_data_path = os.path.join(root_dir, 'data/interim/train_processed.csv')
        logger.info(f'Loading training data from: {train_data_path}')
        train_data = load_data(train_data_path)
        
        # Step 1: Apply TF-IDF feature engineering
        logger.info('\nStep 1: TF-IDF Feature Engineering')
        logger.info('-'*80)
        X_train_tfidf, y_train, vectorizer = apply_tfidf(train_data, params)
        
        # Step 2: Handle class imbalance
        logger.info('\nStep 2: Handling Class Imbalance')
        logger.info('-'*80)
        X_train_resampled, y_train_resampled = handle_imbalance(X_train_tfidf, y_train, params)
        
        # Step 3: Train the LinearSVC model
        logger.info('\nStep 3: Training LinearSVC Model')
        logger.info('-'*80)
        model = train_linearsvc(X_train_resampled, y_train_resampled, params)
        
        # Step 4: Save all artifacts
        logger.info('\nStep 4: Saving Artifacts')
        logger.info('-'*80)
        save_artifacts(vectorizer, model, root_dir)
        
        logger.info('\n' + '='*80)
        logger.info('Model Building Process Completed Successfully!')
        logger.info('='*80)
        logger.info(f'Artifacts saved in: {os.path.join(root_dir, "models")}')
        logger.info('  - tfidf_vectorizer.pkl')
        logger.info('  - linearsvc_model.pkl')
        
    except Exception as e:
        logger.error('='*80)
        logger.error('Failed to complete the model building process: %s', e)
        logger.error('='*80)
        print(f"Error: {e}")
        raise


if __name__ == '__main__':
    main()