import json
import mlflow
import logging
import os
import yaml

# Logging configuration
logger = logging.getLogger('model_registration')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_registration_errors.log')
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
        logger.debug('Parameters loaded from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except Exception as e:
        logger.error('Error loading parameters: %s', e)
        raise


def load_model_info(file_path: str) -> dict:
    """Load the model info from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logger.debug('Model info loaded from %s', file_path)
        logger.debug('Run ID: %s', model_info.get('run_id'))
        logger.debug('Model URI: %s', model_info.get('model_uri'))
        logger.debug('Model Type: %s', model_info.get('model_type'))
        logger.debug('Imbalance Method: %s', model_info.get('imbalance_method'))
        return model_info
    except FileNotFoundError:
        logger.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the model info: %s', e)
        raise


def register_model(model_name: str, model_info: dict, model_version: str, 
                  tags: dict = None) -> dict:
    """
    Register the model to the MLflow Model Registry.
    
    Args:
        model_name: Name for the registered model
        model_info: Dictionary with model information
        model_version: Version string for the model
        tags: Optional dictionary of tags to add to the model version
    
    Returns:
        Dictionary with registration information
    """
    try:
        logger.info('='*80)
        logger.info('Starting Model Registration Process')
        logger.info('='*80)
        
        # Get model URI from model_info
        model_uri = model_info.get('model_uri')
        if not model_uri:
            raise ValueError("model_uri not found in experiment_info.json")
        
        model_type = model_info.get('model_type', 'Unknown')
        imbalance_method = model_info.get('imbalance_method', 'Unknown')
        
        logger.info(f'Model Name: {model_name}')
        logger.info(f'Model Type: {model_type}')
        logger.info(f'Imbalance Method: {imbalance_method}')
        logger.info(f'Model URI: {model_uri}')
        logger.info(f'Version: {model_version}')
        
        # Register the model
        logger.info('\nRegistering model to MLflow Model Registry...')
        registered_model = mlflow.register_model(model_uri, model_name)
        
        logger.info(f'Model registered successfully!')
        logger.info(f'  - Registered Model Name: {registered_model.name}')
        logger.info(f'  - Model Version: {registered_model.version}')
        
        # Get MLflow client
        client = mlflow.tracking.MlflowClient()
        
        # Add tags to the model version
        if tags:
            logger.info('\nAdding tags to model version...')
            for key, value in tags.items():
                client.set_model_version_tag(
                    name=model_name,
                    version=registered_model.version,
                    key=key,
                    value=str(value)
                )
            logger.info(f'Tags added successfully')
            for key, value in tags.items():
                logger.info(f'  - {key}: {value}')
        
        # Add performance metrics as tags
        logger.info('\nAdding performance metrics as tags...')
        test_metrics = model_info.get('test_metrics', {})
        cv_metrics = model_info.get('cross_validation', {})
        per_class_f1 = model_info.get('per_class_f1', {})
        
        metric_tags = {
            'test_accuracy': test_metrics.get('accuracy'),
            'test_f1_weighted': test_metrics.get('f1_weighted'),
            'test_precision_weighted': test_metrics.get('precision_weighted'),
            'test_recall_weighted': test_metrics.get('recall_weighted'),
            'cv_f1_mean': cv_metrics.get('cv_mean'),
            'cv_f1_std': cv_metrics.get('cv_std'),
            'f1_positive': per_class_f1.get('positive'),
            'f1_neutral': per_class_f1.get('neutral'),
            'f1_negative': per_class_f1.get('negative')
        }
        
        for key, value in metric_tags.items():
            if value is not None:
                client.set_model_version_tag(
                    name=model_name,
                    version=registered_model.version,
                    key=key,
                    value=f"{value:.6f}"
                )
        
        logger.info('Performance metrics added:')
        logger.info(f'  - Test Accuracy: {test_metrics.get("accuracy", 0):.6f}')
        logger.info(f'  - Test F1 (weighted): {test_metrics.get("f1_weighted", 0):.6f}')
        logger.info(f'  - CV F1: {cv_metrics.get("cv_mean", 0):.6f} (+/- {cv_metrics.get("cv_std", 0):.6f})')
        
        # Transition the model to "Staging" stage
        logger.info('\nTransitioning model to Staging stage...')
        client.transition_model_version_stage(
            name=model_name,
            version=registered_model.version,
            stage="Staging",
            archive_existing_versions=False
        )
        logger.info(f'Model transitioned to Staging stage')
        
        # Add comprehensive description to model version
        description = (
            f"{model_type} model with {imbalance_method} for sentiment analysis of Reddit comments. "
            f"Performance: Accuracy={test_metrics.get('accuracy', 0):.4f}, "
            f"F1-Score={test_metrics.get('f1_weighted', 0):.4f}, "
            f"CV F1={cv_metrics.get('cv_mean', 0):.4f}±{cv_metrics.get('cv_std', 0):.4f}. "
            f"Per-class F1: Positive={per_class_f1.get('positive', 0):.4f}, "
            f"Neutral={per_class_f1.get('neutral', 0):.4f}, "
            f"Negative={per_class_f1.get('negative', 0):.4f}."
        )
        
        client.update_model_version(
            name=model_name,
            version=registered_model.version,
            description=description
        )
        logger.info(f'Model description updated')
        
        # Prepare registration info
        registration_info = {
            'model_name': model_name,
            'model_version': registered_model.version,
            'model_uri': model_uri,
            'model_type': model_type,
            'imbalance_method': imbalance_method,
            'stage': 'Staging',
            'run_id': model_info.get('run_id'),
            'test_metrics': test_metrics,
            'cross_validation': cv_metrics,
            'per_class_f1': per_class_f1,
            'tags': tags or {}
        }
        
        logger.info('\n' + '='*80)
        logger.info('Model Registration Completed Successfully!')
        logger.info('='*80)
        logger.info(f'Model Name: {model_name}')
        logger.info(f'Model Type: {model_type}')
        logger.info(f'Imbalance Method: {imbalance_method}')
        logger.info(f'Version: {registered_model.version}')
        logger.info(f'Stage: Staging')
        logger.info(f'\nPerformance Metrics:')
        logger.info(f'  Test Accuracy: {test_metrics.get("accuracy", 0):.6f} ({test_metrics.get("accuracy", 0)*100:.2f}%)')
        logger.info(f'  Test F1-Score (weighted): {test_metrics.get("f1_weighted", 0):.6f}')
        logger.info(f'  Test Precision (weighted): {test_metrics.get("precision_weighted", 0):.6f}')
        logger.info(f'  Test Recall (weighted): {test_metrics.get("recall_weighted", 0):.6f}')
        logger.info(f'  CV F1-Score: {cv_metrics.get("cv_mean", 0):.6f} (+/- {cv_metrics.get("cv_std", 0):.6f})')
        logger.info(f'\nPer-Class F1-Scores:')
        logger.info(f'  Positive: {per_class_f1.get("positive", 0):.6f}')
        logger.info(f'  Neutral: {per_class_f1.get("neutral", 0):.6f}')
        logger.info(f'  Negative (Minority): {per_class_f1.get("negative", 0):.6f}')
        logger.info('='*80)
        
        return registration_info
        
    except Exception as e:
        logger.error('Error during model registration: %s', e)
        raise


def save_registry_info(registry_info: dict, root_dir: str) -> str:
    """Save model registry information to JSON file."""
    try:
        reports_dir = os.path.join(root_dir, 'reports')
        os.makedirs(reports_dir, exist_ok=True)
        
        registry_path = os.path.join(reports_dir, 'model_registry_info.json')
        
        with open(registry_path, 'w') as f:
            json.dump(registry_info, f, indent=4)
        
        logger.info(f'\nRegistry info saved: {registry_path}')
        return registry_path
        
    except Exception as e:
        logger.error('Error saving registry info: %s', e)
        raise


def promote_to_production(model_name: str, version: str, client) -> None:
    """
    Promote a model version to Production stage.
    
    Args:
        model_name: Name of the registered model
        version: Version to promote
        client: MLflow tracking client
    """
    try:
        logger.info('\n' + '='*80)
        logger.info('OPTIONAL: Promoting Model to Production')
        logger.info('='*80)
        logger.info('To promote this model to Production, run:')
        logger.info(f'  mlflow models transition {model_name} --version {version} --stage Production')
        logger.info('\nOr use the MLflow UI:')
        logger.info('  1. Navigate to the Models page')
        logger.info(f'  2. Select model: {model_name}')
        logger.info(f'  3. Click on version {version}')
        logger.info('  4. Click "Stage: Staging" dropdown')
        logger.info('  5. Select "Transition to -> Production"')
        logger.info('='*80)
        
    except Exception as e:
        logger.error('Error displaying promotion instructions: %s', e)


def get_root_directory() -> str:
    """Get the root directory."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, '../../'))


def main():
    try:
        logger.info('='*80)
        logger.info('MLflow Model Registration - LightGBM + ADASYN')
        logger.info('='*80)
        
        # Get root directory
        root_dir = get_root_directory()
        
        # Load parameters
        params_path = os.path.join(root_dir, 'params.yaml')
        params = load_params(params_path)
        
        # Set up MLflow
        mlflow_params = params['mlflow']
        mlflow.set_tracking_uri(mlflow_params['tracking_uri'])
        
        logger.info(f'MLflow Tracking URI: {mlflow_params["tracking_uri"]}')
        
        # Load model info
        model_info_path = os.path.join(root_dir, 'reports/experiment_info.json')
        model_info = load_model_info(model_info_path)
        
        # Get deployment parameters
        deployment_params = params['deployment']
        model_name = deployment_params.get('model_name', 'final_lightgbm_adasyn_model')
        model_version_str = deployment_params.get('model_version', '1.0.0')
        
        # Get model type and imbalance method from model_info
        model_type = model_info.get('model_type', 'LightGBM')
        imbalance_method = model_info.get('imbalance_method', 'ADASYN')
        
        # Prepare tags
        tags = {
            'model_type': model_type,
            'version': model_version_str,
            'task': 'sentiment_analysis',
            'dataset': 'reddit_comments',
            'preprocessing': f'tfidf_{imbalance_method.lower()}',
            'imbalance_method': imbalance_method,
            'pipeline': 'dvc',
            'experiment': 'experiment_8',
            'production_ready': str(deployment_params.get('production_ready', True)),
            'feature_engineering': 'tfidf_bigrams_1000',
            'n_classes': '3',
            'best_model': 'true'
        }
        
        # Register the model
        registry_info = register_model(
            model_name=model_name,
            model_info=model_info,
            model_version=model_version_str,
            tags=tags
        )
        
        # Save registry information
        save_registry_info(registry_info, root_dir)
        
        # Get MLflow client
        client = mlflow.tracking.MlflowClient()
        
        # Show promotion instructions
        promote_to_production(model_name, registry_info['model_version'], client)
        
        logger.info('\n' + '='*80)
        logger.info('NEXT STEPS')
        logger.info('='*80)
        logger.info('1. Review the model in MLflow UI')
        logger.info('2. Test the model in Staging environment')
        logger.info('3. If satisfied, promote to Production')
        logger.info('4. Deploy using the model URI from MLflow')
        logger.info('\nModel is ready for deployment!')
        logger.info('='*80)
        
    except Exception as e:
        logger.error('='*80)
        logger.error('Failed to complete the model registration process: %s', e)
        logger.error('='*80)
        print(f"Error: {e}")
        raise


if __name__ == '__main__':
    main()