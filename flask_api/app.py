import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend before importing pyplot

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import io
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import mlflow
import mlflow.sklearn
import numpy as np
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from mlflow.tracking import MlflowClient
import matplotlib.dates as mdates
import pickle
import logging



app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# CONFIGURATION
MLFLOW_TRACKING_URI = "http://ec2-54-211-18-166.compute-1.amazonaws.com:5000/"
MODEL_NAME = "final_lightgbm_adasyn_model"  # Updated for LightGBM model
MODEL_STAGE = "Staging"  # Change to "Production" after promotion

# Sentiment label mapping (matches your training)
SENTIMENT_LABELS = {
    0: "Neutral",
    1: "Positive", 
    2: "Negative"
}

# Reverse mapping for API responses (YouTube convention: -1, 0, 1)
SENTIMENT_TO_YOUTUBE = {
    0: 0,   # Neutral → 0
    1: 1,   # Positive → 1
    2: -1   # Negative → -1
}


# PREPROCESSING FUNCTION
def preprocess_comment(comment):
    """
    Apply preprocessing transformations to a comment.
    Must match the preprocessing used during training.
    """
    try:
        # Convert to lowercase
        comment = comment.lower()

        # Remove trailing and leading whitespaces
        comment = comment.strip()

        # Remove newline characters
        comment = re.sub(r'\n', ' ', comment)
        
        # Remove multiple spaces
        comment = re.sub(r'\s+', ' ', comment)

        # Remove non-alphanumeric characters, except punctuation
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)

        # Remove stopwords but retain important ones for sentiment analysis
        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet', 'nor', 'neither'}
        comment = ' '.join([word for word in comment.split() if word not in stop_words])

        # Lemmatize the words
        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])
        
        # Final cleanup
        comment = ' '.join(comment.split())

        return comment
    except Exception as e:
        logger.error(f"Error in preprocessing comment: {e}")
        return comment



# MODEL LOADING FROM MLFLOW REGISTRY
def load_model_from_registry(model_name, stage_or_version):
   
    try:
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        logger.info(f"MLflow Tracking URI: {MLFLOW_TRACKING_URI}")
        
        client = MlflowClient()
        
        # Determine if stage or version
        if stage_or_version in ["Staging", "Production", "Archived", "None"]:
            # Load by stage
            model_uri = f"models:/{model_name}/{stage_or_version}"
            logger.info(f"Loading model from stage: {stage_or_version}")
        else:
            # Load by version number
            model_uri = f"models:/{model_name}/{stage_or_version}"
            logger.info(f"Loading model version: {stage_or_version}")
        
        logger.info(f"Model URI: {model_uri}")
        
        # Load the model
        model = mlflow.sklearn.load_model(model_uri)
        logger.info("✓ LightGBM model loaded successfully from MLflow Registry")
        
        # Get model version details
        if stage_or_version in ["Staging", "Production", "Archived", "None"]:
            model_versions = client.get_latest_versions(model_name, stages=[stage_or_version])
            if model_versions:
                model_version = model_versions[0]
                run_id = model_version.run_id
                version_number = model_version.version
            else:
                raise ValueError(f"No model found in {stage_or_version} stage")
        else:
            model_version = client.get_model_version(model_name, stage_or_version)
            run_id = model_version.run_id
            version_number = model_version.version
        
        logger.info(f"Model version: {version_number}")
        logger.info(f"Run ID: {run_id}")
        
        # Download the vectorizer artifact from the same run
        artifact_path = f"runs:/{run_id}/tfidf_vectorizer.pkl"
        logger.info(f"Loading vectorizer from: {artifact_path}")
        
        # Download artifact to local path
        local_path = mlflow.artifacts.download_artifacts(artifact_uri=artifact_path)
        
        # Load vectorizer
        with open(local_path, 'rb') as f:
            vectorizer = pickle.load(f)
        
        logger.info("✓ TF-IDF vectorizer loaded successfully")
        
        # Get model tags/metrics
        model_info = {
            "model_name": model_name,
            "version": version_number,
            "stage": stage_or_version,
            "run_id": run_id,
            "model_type": "LightGBM",
            "imbalance_method": "ADASYN"
        }
        
        try:
            tags = {tag.key: tag.value for tag in model_version.tags}
            if 'test_accuracy' in tags:
                model_info['accuracy'] = float(tags['test_accuracy'])
                logger.info(f"Model Accuracy: {tags['test_accuracy']}")
            if 'test_f1_weighted' in tags:
                model_info['f1_score'] = float(tags['test_f1_weighted'])
                logger.info(f"Model F1-Score: {tags['test_f1_weighted']}")
            if 'f1_positive' in tags:
                model_info['f1_positive'] = float(tags['f1_positive'])
            if 'f1_neutral' in tags:
                model_info['f1_neutral'] = float(tags['f1_neutral'])
            if 'f1_negative' in tags:
                model_info['f1_negative'] = float(tags['f1_negative'])
        except Exception as e:
            logger.warning(f"Could not retrieve model tags: {e}")
        
        return model, vectorizer, model_info
        
    except Exception as e:
        logger.error(f"Error loading model from registry: {e}")
        raise



# INITIALIZE MODEL AT STARTUP
logger.info("="*80)
logger.info("🚀 Initializing YouTube Sentiment Analysis API")
logger.info("="*80)

try:
    model, vectorizer, model_info = load_model_from_registry(MODEL_NAME, MODEL_STAGE)
    logger.info("✓ Model and vectorizer loaded successfully!")
    logger.info(f"Model Type: {model_info.get('model_type', 'Unknown')}")
    logger.info(f"Model Version: {model_info.get('version', 'Unknown')}")
    if 'accuracy' in model_info:
        logger.info(f"Model Accuracy: {model_info['accuracy']:.4f}")
    logger.info("="*80)
except Exception as e:
    logger.error(f"❌ Failed to load model: {e}")
    logger.error("Application will not be able to make predictions!")
    model = None
    vectorizer = None
    model_info = {}


# API ENDPOINTS

@app.route('/')
def home():
    """Health check endpoint."""
    status = {
        "status": "running",
        "service": "YouTube Comment Sentiment Analysis API",
        "model_loaded": model is not None,
        "vectorizer_loaded": vectorizer is not None,
        "model_name": MODEL_NAME,
        "model_stage": MODEL_STAGE,
        "model_type": model_info.get('model_type', 'Unknown'),
        "model_version": model_info.get('version', 'Unknown'),
        "mlflow_tracking_uri": MLFLOW_TRACKING_URI
    }
    
    if 'accuracy' in model_info:
        status['model_accuracy'] = f"{model_info['accuracy']:.2%}"
    
    return jsonify(status)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for CI/CD monitoring."""
    return jsonify({
        "status": "healthy",
        "service": "youtube-sentiment-analyzer",
        "version": "1.0.0",
        "model_loaded": model is not None,
        "vectorizer_loaded": vectorizer is not None
    }), 200

@app.route('/model_info', methods=['GET'])
def get_model_info():
    """Get detailed information about the loaded model."""
    try:
        if model is None:
            return jsonify({"error": "Model not loaded"}), 500
        
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient()
        
        # Get latest model version info
        model_versions = client.get_latest_versions(MODEL_NAME, stages=[MODEL_STAGE])
        
        if not model_versions:
            return jsonify({"error": "Model version not found"}), 404
        
        model_version = model_versions[0]
        
        info = {
            "model_name": MODEL_NAME,
            "model_type": "LightGBM",
            "imbalance_method": "ADASYN",
            "version": model_version.version,
            "stage": model_version.current_stage,
            "run_id": model_version.run_id,
            "creation_timestamp": model_version.creation_timestamp,
            "last_updated_timestamp": model_version.last_updated_timestamp,
            "description": model_version.description,
            "tags": {tag.key: tag.value for tag in model_version.tags},
            "sentiment_labels": SENTIMENT_LABELS
        }
        
        return jsonify(info)
        
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/predict', methods=['POST'])
def predict():

    if model is None or vectorizer is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    data = request.json
    comments = data.get('comments')
    
    logger.info(f"Received prediction request for {len(comments) if comments else 0} comments")
    
    if not comments:
        return jsonify({"error": "No comments provided"}), 400

    try:
        # Preprocess each comment
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]
        
        # Transform comments using the vectorizer
        transformed_comments = vectorizer.transform(preprocessed_comments)
        
        # Make predictions
        predictions = model.predict(transformed_comments)
        
        # Get prediction probabilities for confidence scores
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(transformed_comments)
            confidences = np.max(probabilities, axis=1)
        else:
            confidences = [1.0] * len(predictions)  # Default confidence
        
        logger.info(f"✓ Predictions completed: {len(predictions)} sentiments predicted")
        
        # Build response with YouTube convention (-1, 0, 1)
        response = []
        for comment, pred, conf in zip(comments, predictions, confidences):
            pred_int = int(pred)
            youtube_sentiment = SENTIMENT_TO_YOUTUBE[pred_int]
            
            response.append({
                "comment": comment,
                "sentiment": youtube_sentiment,  # YouTube format: -1, 0, 1
                "sentiment_label": SENTIMENT_LABELS[pred_int],
                "confidence": float(conf)
            })
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.route('/predict_with_timestamps', methods=['POST'])
def predict_with_timestamps():

    if model is None or vectorizer is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    data = request.json
    comments_data = data.get('comments')
    
    if not comments_data:
        return jsonify({"error": "No comments provided"}), 400

    try:
        comments = [item['text'] for item in comments_data]
        timestamps = [item['timestamp'] for item in comments_data]

        # Preprocess comments
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]
        
        # Transform and predict
        transformed_comments = vectorizer.transform(preprocessed_comments)
        predictions = model.predict(transformed_comments)
        
        # Get confidence scores
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(transformed_comments)
            confidences = np.max(probabilities, axis=1)
        else:
            confidences = [1.0] * len(predictions)
        
        logger.info(f"✓ Predicted sentiments for {len(predictions)} comments with timestamps")
        
        # Build response
        response = []
        for comment, pred, conf, timestamp in zip(comments, predictions, confidences, timestamps):
            pred_int = int(pred)
            youtube_sentiment = SENTIMENT_TO_YOUTUBE[pred_int]
            
            response.append({
                "comment": comment,
                "sentiment": youtube_sentiment,
                "sentiment_label": SENTIMENT_LABELS[pred_int],
                "confidence": float(conf),
                "timestamp": timestamp
            })
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Prediction with timestamps failed: {e}")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.route('/batch_predict', methods=['POST'])
def batch_predict():

    if model is None or vectorizer is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    data = request.json
    comments = data.get('comments')
    
    if not comments:
        return jsonify({"error": "No comments provided"}), 400

    try:
        # Preprocess and predict
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]
        transformed_comments = vectorizer.transform(preprocessed_comments)
        predictions = model.predict(transformed_comments)
        
        # Calculate statistics
        sentiment_counts = {
            "positive": int(np.sum(predictions == 1)),
            "neutral": int(np.sum(predictions == 0)),
            "negative": int(np.sum(predictions == 2))
        }
        
        total = len(predictions)
        sentiment_percentages = {
            "positive": (sentiment_counts["positive"] / total * 100),
            "neutral": (sentiment_counts["neutral"] / total * 100),
            "negative": (sentiment_counts["negative"] / total * 100)
        }
        
        # Overall sentiment
        if sentiment_counts["positive"] > sentiment_counts["negative"]:
            overall_sentiment = "Positive"
        elif sentiment_counts["negative"] > sentiment_counts["positive"]:
            overall_sentiment = "Negative"
        else:
            overall_sentiment = "Neutral"
        
        logger.info(f"✓ Batch prediction: {total} comments analyzed")
        
        response = {
            "total_comments": total,
            "sentiment_counts": sentiment_counts,
            "sentiment_percentages": sentiment_percentages,
            "overall_sentiment": overall_sentiment,
            "predictions": [
                {
                    "comment": comment,
                    "sentiment": SENTIMENT_TO_YOUTUBE[int(pred)],
                    "sentiment_label": SENTIMENT_LABELS[int(pred)]
                }
                for comment, pred in zip(comments, predictions)
            ]
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        return jsonify({"error": f"Batch prediction failed: {str(e)}"}), 500


@app.route('/generate_chart', methods=['POST'])
def generate_chart():
    """Generate a pie chart for sentiment distribution."""
    try:
        data = request.get_json()
        sentiment_counts = data.get('sentiment_counts')
        
        if not sentiment_counts:
            return jsonify({"error": "No sentiment counts provided"}), 400

        # Map YouTube format to labels
        labels = ['Positive', 'Neutral', 'Negative']
        sizes = [
            int(sentiment_counts.get('1', 0)),   # Positive
            int(sentiment_counts.get('0', 0)),   # Neutral
            int(sentiment_counts.get('-1', 0))   # Negative
        ]
        
        if sum(sizes) == 0:
            return jsonify({"error": "No sentiment data to display"}), 400
        
        colors = ['#36A2EB', '#C9CBCF', '#FF6384']  # Blue, Gray, Red

        # Generate pie chart
        plt.figure(figsize=(10, 8))
        plt.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            startangle=140,
            textprops={'fontsize': 14, 'weight': 'bold'},
            wedgeprops={'edgecolor': 'white', 'linewidth': 2}
        )
        plt.title('YouTube Comment Sentiment Distribution', fontsize=16, fontweight='bold', pad=20)
        plt.axis('equal')

        # Save to BytesIO
        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG', dpi=150, bbox_inches='tight', facecolor='white')
        img_io.seek(0)
        plt.close()

        logger.info("✓ Sentiment chart generated")
        return send_file(img_io, mimetype='image/png')
        
    except Exception as e:
        logger.error(f"Chart generation failed: {e}")
        return jsonify({"error": f"Chart generation failed: {str(e)}"}), 500


@app.route('/generate_wordcloud', methods=['POST'])
def generate_wordcloud():
    """Generate a word cloud from comments (separated by sentiment)."""
    try:
        data = request.get_json()
        comments = data.get('comments')
        sentiment_filter = data.get('sentiment_filter', 'all')  # 'all', 'positive', 'negative', 'neutral'

        if not comments:
            return jsonify({"error": "No comments provided"}), 400

        # Preprocess comments
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]

        # Filter by sentiment if requested
        if sentiment_filter != 'all':
            # You would need to pass predictions or filter on client side
            pass

        # Combine all comments
        text = ' '.join(preprocessed_comments)
        
        if not text.strip():
            return jsonify({"error": "No valid text to generate word cloud"}), 400

        # Generate word cloud
        wordcloud = WordCloud(
            width=1600,
            height=800,
            background_color='white',
            colormap='viridis',
            stopwords=set(stopwords.words('english')),
            collocations=False,
            max_words=150,
            relative_scaling=0.5,
            min_font_size=10
        ).generate(text)

        # Save to BytesIO
        plt.figure(figsize=(16, 8))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud - YouTube Comments', fontsize=18, fontweight='bold', pad=20)
        plt.tight_layout(pad=0)
        
        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG', dpi=150, bbox_inches='tight', facecolor='white')
        img_io.seek(0)
        plt.close()

        logger.info("✓ Word cloud generated")
        return send_file(img_io, mimetype='image/png')
        
    except Exception as e:
        logger.error(f"Word cloud generation failed: {e}")
        return jsonify({"error": f"Word cloud generation failed: {str(e)}"}), 500


@app.route('/generate_trend_graph', methods=['POST'])
def generate_trend_graph():
    """Generate a trend graph showing sentiment over time."""
    try:
        data = request.get_json()
        sentiment_data = data.get('sentiment_data')

        if not sentiment_data:
            return jsonify({"error": "No sentiment data provided"}), 400

        # Convert to DataFrame
        df = pd.DataFrame(sentiment_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        df['sentiment'] = df['sentiment'].astype(int)

        # Map sentiment values
        sentiment_labels = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}

        # Resample monthly
        monthly_counts = df.resample('M')['sentiment'].value_counts().unstack(fill_value=0)
        monthly_totals = monthly_counts.sum(axis=1)
        monthly_percentages = (monthly_counts.T / monthly_totals).T * 100

        # Ensure all sentiment columns exist
        for sentiment_value in [-1, 0, 1]:
            if sentiment_value not in monthly_percentages.columns:
                monthly_percentages[sentiment_value] = 0

        monthly_percentages = monthly_percentages[[-1, 0, 1]]

        # Plot
        plt.figure(figsize=(16, 8))
        colors = {-1: '#FF6384', 0: '#C9CBCF', 1: '#36A2EB'}

        for sentiment_value in [-1, 0, 1]:
            plt.plot(
                monthly_percentages.index,
                monthly_percentages[sentiment_value],
                marker='o',
                linestyle='-',
                linewidth=2.5,
                markersize=8,
                label=sentiment_labels[sentiment_value],
                color=colors[sentiment_value]
            )

        plt.title('YouTube Comment Sentiment Trend Over Time', fontsize=18, fontweight='bold', pad=20)
        plt.xlabel('Month', fontsize=14, fontweight='bold')
        plt.ylabel('Percentage of Comments (%)', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.xticks(rotation=45)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))
        plt.legend(fontsize=12, loc='best')
        plt.tight_layout()

        # Save to BytesIO
        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG', dpi=150, bbox_inches='tight', facecolor='white')
        img_io.seek(0)
        plt.close()

        logger.info("✓ Trend graph generated")
        return send_file(img_io, mimetype='image/png')
        
    except Exception as e:
        logger.error(f"Trend graph generation failed: {e}")
        return jsonify({"error": f"Trend graph generation failed: {str(e)}"}), 500


# ERROR HANDLERS

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500


# RUN APPLICATION

if __name__ == '__main__':
    logger.info("="*80)
    logger.info("🚀 Starting Flask Application")
    logger.info("="*80)
    logger.info("API Endpoints:")
    logger.info("  GET  /              - Health check")
    logger.info("  GET  /model_info    - Model information")
    logger.info("  POST /predict       - Predict sentiment for comments")
    logger.info("  POST /predict_with_timestamps - Predict with timestamps")
    logger.info("  POST /batch_predict - Batch prediction with stats")
    logger.info("  POST /generate_chart - Generate pie chart")
    logger.info("  POST /generate_wordcloud - Generate word cloud")
    logger.info("  POST /generate_trend_graph - Generate trend graph")
    logger.info("="*80)
    
     # Get port from environment variable or default to 8080
    import os
    port = int(os.getenv('PORT', 8080))
    
    app.run(host='0.0.0.0', port=port, debug=False) 