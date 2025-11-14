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
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MLflow Configuration
MLFLOW_TRACKING_URI = "http://ec2-54-211-18-166.compute-1.amazonaws.com:5000/"
MODEL_NAME = "final_linearsvc_model"  # Name from your Model Registry
MODEL_STAGE = "Staging"  # or "Production" or version number like "1"

# Define the preprocessing function
def preprocess_comment(comment):
    """Apply preprocessing transformations to a comment."""
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


def load_model_from_registry(model_name, stage_or_version):
    """
    Load model and vectorizer from MLflow Model Registry.
    
    Args:
        model_name: Name of the registered model
        stage_or_version: Stage name ("Staging", "Production") or version number ("1", "2", etc.)
    
    Returns:
        Tuple of (model, vectorizer)
    """
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
        logger.info("Model loaded successfully from MLflow Registry")
        
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
        
        logger.info("Vectorizer loaded successfully")
        
        # Log model information
        logger.info(f"Model Name: {model_name}")
        logger.info(f"Model Version: {version_number}")
        logger.info(f"Model Stage: {stage_or_version}")
        
        # Get and log model tags/metrics if available
        try:
            tags = {tag.key: tag.value for tag in model_version.tags}
            if 'test_accuracy' in tags:
                logger.info(f"Model Accuracy: {tags['test_accuracy']}")
            if 'test_f1_weighted' in tags:
                logger.info(f"Model F1-Score: {tags['test_f1_weighted']}")
        except Exception as e:
            logger.warning(f"Could not retrieve model tags: {e}")
        
        return model, vectorizer
        
    except Exception as e:
        logger.error(f"Error loading model from registry: {e}")
        raise


# Initialize the model and vectorizer at startup
logger.info("="*80)
logger.info("Initializing Flask Application")
logger.info("="*80)

try:
    model, vectorizer = load_model_from_registry(MODEL_NAME, MODEL_STAGE)
    logger.info("Model and vectorizer loaded successfully!")
    logger.info("="*80)
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    logger.error("Application will not be able to make predictions!")
    model = None
    vectorizer = None


@app.route('/')
def home():
    """Health check endpoint."""
    status = {
        "status": "running",
        "model_loaded": model is not None,
        "vectorizer_loaded": vectorizer is not None,
        "model_name": MODEL_NAME,
        "model_stage": MODEL_STAGE,
        "mlflow_tracking_uri": MLFLOW_TRACKING_URI
    }
    return jsonify(status)


@app.route('/model_info', methods=['GET'])
def model_info():
    """Get information about the loaded model."""
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
            "version": model_version.version,
            "stage": model_version.current_stage,
            "run_id": model_version.run_id,
            "creation_timestamp": model_version.creation_timestamp,
            "last_updated_timestamp": model_version.last_updated_timestamp,
            "description": model_version.description,
            "tags": {tag.key: tag.value for tag in model_version.tags}
        }
        
        return jsonify(info)
        
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/predict_with_timestamps', methods=['POST'])
def predict_with_timestamps():
    """Predict sentiment for comments with timestamps."""
    if model is None or vectorizer is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    data = request.json
    comments_data = data.get('comments')
    
    if not comments_data:
        return jsonify({"error": "No comments provided"}), 400

    try:
        comments = [item['text'] for item in comments_data]
        timestamps = [item['timestamp'] for item in comments_data]

        # Preprocess each comment before vectorizing
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]
        
        # Transform comments using the vectorizer
        transformed_comments = vectorizer.transform(preprocessed_comments)
        
        # Make predictions
        predictions = model.predict(transformed_comments).tolist()
        
        # Convert predictions to integers
        predictions = [int(pred) for pred in predictions]
        
        logger.info(f"Predicted sentiments for {len(predictions)} comments with timestamps")
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
    
    # Return the response with original comments, predicted sentiments, and timestamps
    response = [
        {"comment": comment, "sentiment": sentiment, "timestamp": timestamp} 
        for comment, sentiment, timestamp in zip(comments, predictions, timestamps)
    ]
    return jsonify(response)


@app.route('/predict', methods=['POST'])
def predict():
    """Predict sentiment for a list of comments."""
    if model is None or vectorizer is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    data = request.json
    comments = data.get('comments')
    
    logger.info(f"Received {len(comments) if comments else 0} comments for prediction")
    
    if not comments:
        return jsonify({"error": "No comments provided"}), 400

    try:
        # Preprocess each comment before vectorizing
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]
        
        # Transform comments using the vectorizer
        transformed_comments = vectorizer.transform(preprocessed_comments)
        
        # Make predictions
        predictions = model.predict(transformed_comments).tolist()
        
        # Convert predictions to integers
        predictions = [int(pred) for pred in predictions]
        
        logger.info(f"Predictions completed: {len(predictions)} sentiments predicted")
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
    
    # Return the response with original comments and predicted sentiments
    response = [
        {"comment": comment, "sentiment": sentiment} 
        for comment, sentiment in zip(comments, predictions)
    ]
    return jsonify(response)


@app.route('/generate_chart', methods=['POST'])
def generate_chart():
    """Generate a pie chart for sentiment distribution."""
    try:
        data = request.get_json()
        sentiment_counts = data.get('sentiment_counts')
        
        if not sentiment_counts:
            return jsonify({"error": "No sentiment counts provided"}), 400

        # Prepare data for the pie chart
        labels = ['Positive', 'Neutral', 'Negative']
        sizes = [
            int(sentiment_counts.get('1', 0)),
            int(sentiment_counts.get('0', 0)),
            int(sentiment_counts.get('-1', 0))
        ]
        
        if sum(sizes) == 0:
            return jsonify({"error": "No sentiment data to display"}), 400
        
        colors = ['#36A2EB', '#C9CBCF', '#FF6384']  # Blue, Gray, Red

        # Generate the pie chart
        plt.figure(figsize=(8, 8))
        plt.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            startangle=140,
            textprops={'color': 'w', 'fontsize': 12}
        )
        plt.title('Sentiment Distribution', fontsize=14, color='white')
        plt.axis('equal')

        # Save the chart to a BytesIO object
        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG', transparent=True, bbox_inches='tight')
        img_io.seek(0)
        plt.close()

        logger.info("Sentiment chart generated successfully")
        return send_file(img_io, mimetype='image/png')
        
    except Exception as e:
        logger.error(f"Error in /generate_chart: {e}")
        return jsonify({"error": f"Chart generation failed: {str(e)}"}), 500


@app.route('/generate_wordcloud', methods=['POST'])
def generate_wordcloud():
    """Generate a word cloud from comments."""
    try:
        data = request.get_json()
        comments = data.get('comments')

        if not comments:
            return jsonify({"error": "No comments provided"}), 400

        # Preprocess comments
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]

        # Combine all comments into a single string
        text = ' '.join(preprocessed_comments)
        
        if not text.strip():
            return jsonify({"error": "No valid text to generate word cloud"}), 400

        # Generate the word cloud
        wordcloud = WordCloud(
            width=1200,
            height=600,
            background_color='black',
            colormap='Blues',
            stopwords=set(stopwords.words('english')),
            collocations=False,
            max_words=100
        ).generate(text)

        # Save the word cloud to a BytesIO object
        img_io = io.BytesIO()
        wordcloud.to_image().save(img_io, format='PNG')
        img_io.seek(0)

        logger.info("Word cloud generated successfully")
        return send_file(img_io, mimetype='image/png')
        
    except Exception as e:
        logger.error(f"Error in /generate_wordcloud: {e}")
        return jsonify({"error": f"Word cloud generation failed: {str(e)}"}), 500


@app.route('/generate_trend_graph', methods=['POST'])
def generate_trend_graph():
    """Generate a trend graph showing sentiment over time."""
    try:
        data = request.get_json()
        sentiment_data = data.get('sentiment_data')

        if not sentiment_data:
            return jsonify({"error": "No sentiment data provided"}), 400

        # Convert sentiment_data to DataFrame
        df = pd.DataFrame(sentiment_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Set the timestamp as the index
        df.set_index('timestamp', inplace=True)

        # Ensure the 'sentiment' column is numeric
        df['sentiment'] = df['sentiment'].astype(int)

        # Map sentiment values to labels
        sentiment_labels = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}

        # Resample the data over monthly intervals and count sentiments
        monthly_counts = df.resample('M')['sentiment'].value_counts().unstack(fill_value=0)

        # Calculate total counts per month
        monthly_totals = monthly_counts.sum(axis=1)

        # Calculate percentages
        monthly_percentages = (monthly_counts.T / monthly_totals).T * 100

        # Ensure all sentiment columns are present
        for sentiment_value in [-1, 0, 1]:
            if sentiment_value not in monthly_percentages.columns:
                monthly_percentages[sentiment_value] = 0

        # Sort columns by sentiment value
        monthly_percentages = monthly_percentages[[-1, 0, 1]]

        # Plotting
        plt.figure(figsize=(14, 7))

        colors = {
            -1: '#FF6384',  # Red - Negative sentiment
            0: '#C9CBCF',   # Gray - Neutral sentiment
            1: '#36A2EB'    # Blue - Positive sentiment
        }

        for sentiment_value in [-1, 0, 1]:
            plt.plot(
                monthly_percentages.index,
                monthly_percentages[sentiment_value],
                marker='o',
                linestyle='-',
                linewidth=2,
                markersize=6,
                label=sentiment_labels[sentiment_value],
                color=colors[sentiment_value]
            )

        plt.title('Monthly Sentiment Trend Over Time', fontsize=16, fontweight='bold')
        plt.xlabel('Month', fontsize=12)
        plt.ylabel('Percentage of Comments (%)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)

        # Format the x-axis dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))

        plt.legend(fontsize=11)
        plt.tight_layout()

        # Save the trend graph to a BytesIO object
        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG', dpi=150, bbox_inches='tight')
        img_io.seek(0)
        plt.close()

        logger.info("Trend graph generated successfully")
        return send_file(img_io, mimetype='image/png')
        
    except Exception as e:
        logger.error(f"Error in /generate_trend_graph: {e}")
        return jsonify({"error": f"Trend graph generation failed: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)