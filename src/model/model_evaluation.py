import numpy as np
import pandas as pd
import pickle
import logging
import yaml
import mlflow
import mlflow.sklearn
import os
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from mlflow.models import infer_signature

# Logging configuration
logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_evaluation_errors.log')
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
    except Exception as e:
        logger.error('Error loading parameters from %s: %s', params_path, e)
        raise


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        df.fillna('', inplace=True)
        logger.debug('Data loaded from %s with shape %s', file_path, df.shape)
        return df
    except Exception as e:
        logger.error('Error loading data from %s: %s', file_path, e)
        raise


def load_pickle(file_path: str):
    """Load pickled object."""
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        logger.debug('Data loaded from %s', file_path)
        return data
    except Exception as e:
        logger.error('Error loading from %s: %s', file_path, e)
        raise


def evaluate_model(model, X_test, y_test) -> tuple:
    """
    Evaluate the model and return comprehensive metrics.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
    
    Returns:
        Tuple of (metrics_dict, classification_report, confusion_matrix, predictions)
    """
    try:
        logger.info('Starting model evaluation...')
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate overall metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision_macro = precision_score(y_test, y_pred, average='macro')
        precision_weighted = precision_score(y_test, y_pred, average='weighted')
        recall_macro = recall_score(y_test, y_pred, average='macro')
        recall_weighted = recall_score(y_test, y_pred, average='weighted')
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        
        # Per-class metrics
        precision_per_class = precision_score(y_test, y_pred, average=None)
        recall_per_class = recall_score(y_test, y_pred, average=None)
        f1_per_class = f1_score(y_test, y_pred, average=None)
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        logger.info('Model evaluation completed')
        logger.info(f'  - Accuracy: {accuracy:.6f}')
        logger.info(f'  - Weighted F1-Score: {f1_weighted:.6f}')
        
        # Compile metrics
        metrics = {
            'accuracy': float(accuracy),
            'precision_macro': float(precision_macro),
            'precision_weighted': float(precision_weighted),
            'recall_macro': float(recall_macro),
            'recall_weighted': float(recall_weighted),
            'f1_macro': float(f1_macro),
            'f1_weighted': float(f1_weighted),
            'precision_per_class': [float(x) for x in precision_per_class],
            'recall_per_class': [float(x) for x in recall_per_class],
            'f1_per_class': [float(x) for x in f1_per_class]
        }
        
        return metrics, report, cm, y_pred
        
    except Exception as e:
        logger.error('Error during model evaluation: %s', e)
        raise


def perform_cross_validation(model, X_train, y_train, params: dict) -> dict:
    """
    Perform cross-validation on the training data.
    
    Args:
        model: Trained model
        X_train: Training features
        y_train: Training labels
        params: CV parameters
    
    Returns:
        Dictionary with CV metrics
    """
    try:
        logger.info('Starting cross-validation...')
        
        eval_params = params['model_evaluation']
        cv_folds = eval_params['cv_folds']
        cv_shuffle = eval_params['cv_shuffle']
        cv_random_state = eval_params['cv_random_state']
        
        # Stratified K-Fold
        cv = StratifiedKFold(
            n_splits=cv_folds,
            shuffle=cv_shuffle,
            random_state=cv_random_state
        )
        
        # Cross-validation scores
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1_weighted')
        
        cv_metrics = {
            'cv_scores': [float(x) for x in cv_scores],
            'cv_mean': float(cv_scores.mean()),
            'cv_std': float(cv_scores.std()),
            'cv_min': float(cv_scores.min()),
            'cv_max': float(cv_scores.max())
        }
        
        logger.info(f'Cross-validation completed')
        logger.info(f'  - CV F1-Score: {cv_metrics["cv_mean"]:.6f} (+/- {cv_metrics["cv_std"] * 2:.6f})')
        
        return cv_metrics
        
    except Exception as e:
        logger.error('Error during cross-validation: %s', e)
        raise


def create_visualizations(cm, y_test, y_pred, classes, model, vectorizer, root_dir: str) -> list:
    """
    Create and save visualization plots including feature importance.
    
    Args:
        cm: Confusion matrix
        y_test: True labels
        y_pred: Predicted labels
        classes: Unique class labels
        model: Trained LightGBM model
        vectorizer: TF-IDF vectorizer
        root_dir: Root directory path
    
    Returns:
        List of plot file paths
    """
    try:
        logger.info('Creating visualizations...')
        
        # Create plots directory
        plots_dir = os.path.join(root_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        plot_paths = []
        
        # 1. Confusion Matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.tight_layout()
        cm_path = os.path.join(plots_dir, 'confusion_matrix.png')
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_paths.append(cm_path)
        logger.info(f'Confusion matrix saved: {cm_path}')
        
        # 2. Normalized Confusion Matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='RdYlGn', 
                   cbar_kws={'label': 'Percentage'})
        plt.title('Normalized Confusion Matrix', fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.tight_layout()
        cm_norm_path = os.path.join(plots_dir, 'confusion_matrix_normalized.png')
        plt.savefig(cm_norm_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_paths.append(cm_norm_path)
        logger.info(f'Normalized confusion matrix saved: {cm_norm_path}')
        
        # 3. Per-Class Performance
        precision_per_class = precision_score(y_test, y_pred, average=None)
        recall_per_class = recall_score(y_test, y_pred, average=None)
        f1_per_class = f1_score(y_test, y_pred, average=None)
        
        plt.figure(figsize=(10, 6))
        x = np.arange(len(classes))
        width = 0.25
        plt.bar(x - width, precision_per_class, width, label='Precision', alpha=0.8)
        plt.bar(x, recall_per_class, width, label='Recall', alpha=0.8)
        plt.bar(x + width, f1_per_class, width, label='F1-Score', alpha=0.8)
        plt.xlabel('Class', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
        plt.xticks(x, classes)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        perf_path = os.path.join(plots_dir, 'per_class_performance.png')
        plt.savefig(perf_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_paths.append(perf_path)
        logger.info(f'Per-class performance saved: {perf_path}')
        
        # 4. Feature Importance (Top 20) - LightGBM specific
        try:
            feature_importance = model.feature_importances_
            feature_names = vectorizer.get_feature_names_out()
            
            # Get top 20 features
            top_indices = np.argsort(feature_importance)[-20:]
            top_features = [feature_names[i] for i in top_indices]
            top_importance = feature_importance[top_indices]
            
            plt.figure(figsize=(10, 8))
            plt.barh(range(len(top_features)), top_importance, alpha=0.8, color='steelblue')
            plt.yticks(range(len(top_features)), top_features, fontsize=9)
            plt.xlabel('Importance', fontsize=12)
            plt.title('Top 20 Feature Importance', fontsize=14, fontweight='bold')
            plt.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            feat_path = os.path.join(plots_dir, 'feature_importance.png')
            plt.savefig(feat_path, dpi=300, bbox_inches='tight')
            plt.close()
            plot_paths.append(feat_path)
            logger.info(f'Feature importance plot saved: {feat_path}')
        except Exception as e:
            logger.warning(f'Could not create feature importance plot: {e}')
        
        return plot_paths
        
    except Exception as e:
        logger.error('Error creating visualizations: %s', e)
        raise


def save_classification_report(report: dict, root_dir: str) -> str:
    """Save classification report to text file."""
    try:
        reports_dir = os.path.join(root_dir, 'reports')
        os.makedirs(reports_dir, exist_ok=True)
        
        report_path = os.path.join(reports_dir, 'classification_report.txt')
        
        class_names = {
            '0': 'Positive',
            '1': 'Neutral', 
            '2': 'Negative (Minority)'
        }
        
        with open(report_path, 'w') as f:
            f.write("CLASSIFICATION REPORT - LightGBM Model (ADASYN)\n")
            f.write("="*80 + "\n\n")
            
            for label, metrics in report.items():
                if isinstance(metrics, dict):
                    class_name = class_names.get(str(label), str(label))
                    f.write(f"Class {label} - {class_name}:\n")
                    f.write(f"  Precision: {metrics['precision']:.6f}\n")
                    f.write(f"  Recall: {metrics['recall']:.6f}\n")
                    f.write(f"  F1-Score: {metrics['f1-score']:.6f}\n")
                    f.write(f"  Support: {metrics['support']}\n\n")
        
        logger.info(f'Classification report saved: {report_path}')
        return report_path
        
    except Exception as e:
        logger.error('Error saving classification report: %s', e)
        raise


def save_metrics_json(metrics: dict, cv_metrics: dict, root_dir: str) -> str:
    """Save metrics to JSON file for DVC."""
    try:
        metrics_dir = os.path.join(root_dir, 'metrics')
        os.makedirs(metrics_dir, exist_ok=True)
        
        metrics_path = os.path.join(metrics_dir, 'metrics.json')
        
        # Combine all metrics
        all_metrics = {
            'test_accuracy': metrics['accuracy'],
            'test_precision_weighted': metrics['precision_weighted'],
            'test_recall_weighted': metrics['recall_weighted'],
            'test_f1_weighted': metrics['f1_weighted'],
            'cv_f1_mean': cv_metrics['cv_mean'],
            'cv_f1_std': cv_metrics['cv_std']
        }
        
        with open(metrics_path, 'w') as f:
            json.dump(all_metrics, f, indent=4)
        
        logger.info(f'Metrics saved: {metrics_path}')
        return metrics_path
        
    except Exception as e:
        logger.error('Error saving metrics JSON: %s', e)
        raise


def save_experiment_info(run_id: str, model_uri: str, metrics: dict, 
                        cv_metrics: dict, root_dir: str) -> str:
    """Save experiment information for model registration."""
    try:
        reports_dir = os.path.join(root_dir, 'reports')
        os.makedirs(reports_dir, exist_ok=True)
        
        info_path = os.path.join(reports_dir, 'experiment_info.json')
        
        experiment_info = {
            'run_id': run_id,
            'model_uri': model_uri,
            'model_type': 'LightGBM',
            'imbalance_method': 'ADASYN',
            'test_metrics': {
                'accuracy': metrics['accuracy'],
                'f1_weighted': metrics['f1_weighted'],
                'precision_weighted': metrics['precision_weighted'],
                'recall_weighted': metrics['recall_weighted']
            },
            'cross_validation': {
                'cv_mean': cv_metrics['cv_mean'],
                'cv_std': cv_metrics['cv_std']
            },
            'per_class_f1': {
                'positive': metrics['f1_per_class'][0],
                'neutral': metrics['f1_per_class'][1],
                'negative': metrics['f1_per_class'][2]
            }
        }
        
        with open(info_path, 'w') as f:
            json.dump(experiment_info, f, indent=4)
        
        logger.info(f'Experiment info saved: {info_path}')
        logger.info(f'Model URI: {model_uri}')
        return info_path
        
    except Exception as e:
        logger.error('Error saving experiment info: %s', e)
        raise


def get_root_directory() -> str:
    """Get the root directory."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, '../../'))


def main():
    try:
        logger.info('='*80)
        logger.info('Starting Model Evaluation Process - LightGBM + ADASYN')
        logger.info('='*80)
        
        # Get root directory
        root_dir = get_root_directory()
        
        # Load parameters
        params_path = os.path.join(root_dir, 'params.yaml')
        params = load_params(params_path)
        
        # Set up MLflow
        mlflow_params = params['mlflow']
        mlflow.set_tracking_uri(mlflow_params['tracking_uri'])
        mlflow.set_experiment(mlflow_params['experiment_name'])
        
        with mlflow.start_run() as run:
            logger.info(f'MLflow Run ID: {run.info.run_id}')
            
            # Load model and vectorizer
            logger.info('\nLoading model and vectorizer...')
            models_dir = os.path.join(root_dir, 'models')
            model = load_pickle(os.path.join(models_dir, 'lightgbm_model.pkl'))
            vectorizer = load_pickle(os.path.join(models_dir, 'tfidf_vectorizer.pkl'))
            
            # Load processed data
            logger.info('Loading processed data...')
            train_data = load_data(os.path.join(root_dir, 'data/interim/train_processed.csv'))
            test_data = load_data(os.path.join(root_dir, 'data/interim/test_processed.csv'))
            
            # Transform data using vectorizer
            logger.info('Vectorizing text data...')
            X_train = vectorizer.transform(train_data['clean_comment'].values)
            y_train = train_data['category'].values
            X_test = vectorizer.transform(test_data['clean_comment'].values)
            y_test = test_data['category'].values
            
            logger.info(f'Training data shape: {X_train.shape}')
            logger.info(f'Test data shape: {X_test.shape}')
            
            # Log all parameters
            logger.info('\nLogging parameters to MLflow...')
            for section, section_params in params.items():
                if isinstance(section_params, dict):
                    for key, value in section_params.items():
                        if isinstance(value, (int, float, str, bool)):
                            mlflow.log_param(f"{section}.{key}", value)
                        else:
                            mlflow.log_param(f"{section}.{key}", str(value))
                else:
                    mlflow.log_param(section, section_params)
            
            # Evaluate model on test data
            logger.info('\nEvaluating model on test data...')
            metrics, report, cm, y_pred = evaluate_model(model, X_test, y_test)
            
            # Cross-validation on training data
            logger.info('\nPerforming cross-validation on training data...')
            cv_metrics = perform_cross_validation(model, X_train, y_train, params)
            
            # Log metrics to MLflow
            logger.info('\nLogging metrics to MLflow...')
            mlflow.log_metric('test_accuracy', metrics['accuracy'])
            mlflow.log_metric('test_precision_macro', metrics['precision_macro'])
            mlflow.log_metric('test_precision_weighted', metrics['precision_weighted'])
            mlflow.log_metric('test_recall_macro', metrics['recall_macro'])
            mlflow.log_metric('test_recall_weighted', metrics['recall_weighted'])
            mlflow.log_metric('test_f1_macro', metrics['f1_macro'])
            mlflow.log_metric('test_f1_weighted', metrics['f1_weighted'])
            mlflow.log_metric('cv_f1_mean', cv_metrics['cv_mean'])
            mlflow.log_metric('cv_f1_std', cv_metrics['cv_std'])
            
            # Log per-class metrics
            classes = np.unique(y_test)
            class_names = ['positive', 'neutral', 'negative']
            for idx, (class_label, class_name) in enumerate(zip(classes, class_names)):
                mlflow.log_metric(f'test_class_{class_label}_{class_name}_precision', 
                                metrics['precision_per_class'][idx])
                mlflow.log_metric(f'test_class_{class_label}_{class_name}_recall', 
                                metrics['recall_per_class'][idx])
                mlflow.log_metric(f'test_class_{class_label}_{class_name}_f1', 
                                metrics['f1_per_class'][idx])
            
            # Create visualizations (including feature importance)
            logger.info('\nCreating visualizations...')
            plot_paths = create_visualizations(cm, y_test, y_pred, classes, model, vectorizer, root_dir)
            
            # Log plots to MLflow
            for plot_path in plot_paths:
                mlflow.log_artifact(plot_path)
            
            # Save classification report
            logger.info('\nSaving classification report...')
            report_path = save_classification_report(report, root_dir)
            mlflow.log_artifact(report_path)
            
            # Save metrics JSON for DVC
            logger.info('\nSaving metrics JSON for DVC...')
            metrics_path = save_metrics_json(metrics, cv_metrics, root_dir)
            
            # Create input example for signature
            logger.info('\nCreating model signature...')
            sample_size = min(5, X_test.shape[0])
            X_sample = X_test[:sample_size]
            input_example = pd.DataFrame(
                X_sample.toarray(), 
                columns=vectorizer.get_feature_names_out()
            )
            
            # Infer signature
            signature = infer_signature(input_example, model.predict(X_sample))
            
            # Log model with signature
            logger.info('\nLogging model to MLflow...')
            model_info = mlflow.sklearn.log_model(
                model,
                "lightgbm_model",
                signature=signature,
                input_example=input_example
            )
            
            # Log vectorizer as artifact
            mlflow.log_artifact(os.path.join(models_dir, 'tfidf_vectorizer.pkl'))
            
            # Log native LightGBM model if exists
            native_model_path = os.path.join(models_dir, 'lightgbm_model.txt')
            if os.path.exists(native_model_path):
                mlflow.log_artifact(native_model_path)
            
            # Save experiment info for model registration
            logger.info('\nSaving experiment info...')
            
            # Get the full artifact URI (S3 path) - must be AFTER logging model
            artifact_uri = mlflow.get_artifact_uri()
            logger.info(f'Artifact URI: {artifact_uri}')
            
            # Get model URI (full path to model)
            model_uri = model_info.model_uri
            logger.info(f'Model URI: {model_uri}')
            
            info_path = save_experiment_info(
                run.info.run_id, 
                model_uri,
                metrics,
                cv_metrics,
                root_dir
            )
            
            # Set tags
            mlflow.set_tag("model_type", "LightGBM")
            mlflow.set_tag("task", "Sentiment Analysis")
            mlflow.set_tag("dataset", "Reddit Comments")
            mlflow.set_tag("preprocessing", "TF-IDF + ADASYN")
            mlflow.set_tag("pipeline", "DVC")
            mlflow.set_tag("imbalance_method", "adasyn")
            mlflow.set_tag("experiment", "Experiment_8")
            mlflow.set_tag("production_ready", "true")
            
            logger.info('\n' + '='*80)
            logger.info('Model Evaluation Completed Successfully!')
            logger.info('='*80)
            logger.info(f'Model: LightGBM with ADASYN')
            logger.info(f'Test Accuracy: {metrics["accuracy"]:.6f} ({metrics["accuracy"]*100:.2f}%)')
            logger.info(f'Test F1-Score (weighted): {metrics["f1_weighted"]:.6f}')
            logger.info(f'Test F1-Score (macro): {metrics["f1_macro"]:.6f}')
            logger.info(f'CV F1-Score: {cv_metrics["cv_mean"]:.6f} (+/- {cv_metrics["cv_std"]*2:.6f})')
            logger.info('\nPer-Class F1-Scores:')
            logger.info(f'  Positive (0): {metrics["f1_per_class"][0]:.6f}')
            logger.info(f'  Neutral (1): {metrics["f1_per_class"][1]:.6f}')
            logger.info(f'  Negative (2): {metrics["f1_per_class"][2]:.6f}')
            logger.info(f'\nMLflow Run ID: {run.info.run_id}')
            logger.info(f'Model URI: {model_uri}')
            logger.info('='*80)
            
    except Exception as e:
        logger.error('='*80)
        logger.error(f'Failed to complete model evaluation: {e}')
        logger.error('='*80)
        print(f"Error: {e}")
        raise


if __name__ == '__main__':
    main()