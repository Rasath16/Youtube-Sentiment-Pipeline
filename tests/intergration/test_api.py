import pytest
import json
import sys 
import pickle # Used to create the mock vectorizer artifact
from scipy.sparse import csr_matrix # Used by MockVectorizer
from pytest_mock import mocker # Required for mocking

# --- Mock Classes ---
class MockVectorizer:
    """Mock class that mimics the TfidfVectorizer's transform method."""
    def transform(self, comments):
        # Always return a placeholder sparse matrix
        # Uses a consistent shape for testing
        return csr_matrix([[1, 0]] * len(comments))

class MockModel:
    """Mock class for the LightGBM model."""
    def predict(self, features):
        # Predicts [1, 2, 0, 1, 2, ...] 
        # (Internal mapping: 0: Neutral, 1: Positive, 2: Negative)
        return [1, 2, 0, 1, 2] * (len(features.toarray()) // 5 + 1)
    
    def predict_proba(self, features):
        # Simulate high confidence prediction
        return [[0.1, 0.8, 0.1]] * len(features.toarray())

mock_model_info = {
    "version": 99,
    "accuracy": 0.85, 
    "model_type": "LightGBM",
    "stage": "Staging",
    "run_id": "mock_run_id",
}

# --- Fixtures ---

# FIX: Fixture now mocks underlying MLflow dependencies for reliable startup
@pytest.fixture(scope='function')
def client(mocker, tmp_path): 
    
    # 1. Mock MlflowClient and its methods (needed for model_info and artifact path)
    mock_client = mocker.Mock()
    # Ensure get_latest_versions returns a mock object with required properties (version, run_id, tags)
    mock_client.get_latest_versions.return_value = [
        mocker.Mock(
            version=99, 
            run_id='mock_run_id', 
            current_stage='Staging', 
            tags={'test_accuracy': '0.85', 'test_f1_weighted': '0.80'} # Tags needed for model_info extraction
        )
    ]
    mocker.patch('flask_api.app.MlflowClient', return_value=mock_client)

    # 2. Mock model loading (mlflow.sklearn.load_model)
    mocker.patch('flask_api.app.mlflow.sklearn.load_model', return_value=MockModel())
    
    # 3. Mock vectorizer downloading (mlflow.artifacts.download_artifacts)
    # Create a temporary pickle file using tmp_path and dump our MockVectorizer instance
    vectorizer_path = tmp_path / 'tfidf_vectorizer.pkl'
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(MockVectorizer(), f)

    # Mock artifact download to return the path to our dummy file
    mocker.patch('flask_api.app.mlflow.artifacts.download_artifacts', return_value=str(vectorizer_path))

    # 4. Force module reload to run the global initialization (model loading)
    # This guarantees the global code runs with the mocks active.
    if 'flask_api.app' in sys.modules:
        del sys.modules['flask_api.app']
        
    from flask_api.app import app # Force Reload
    
    app.config['TESTING'] = True
    
    with app.test_client() as client:
        yield client 

# --- API Tests ---

def test_health_check(client):
    """Test the /health endpoint"""
    response = client.get('/health')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['status'] == 'healthy'
    assert data['model_loaded'] == True # PASSES: Model load now succeeds due to mocking

def test_home_endpoint(client):
    """Test the root endpoint"""
    response = client.get('/')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['status'] == 'running'
    assert data['model_accuracy'] == '85.00%' # PASSES: model_info is correctly populated

def test_predict_single_comment(client):
    """Test the /predict endpoint with a single comment"""
    comment = "I loved this video, you are the best!"
    response = client.post('/predict', 
                           data=json.dumps({"comments": [comment]}), 
                           content_type='application/json')
    
    assert response.status_code == 200 # PASSES: Prediction runs successfully
    data = json.loads(response.data)
    
    assert len(data) == 1
    assert data[0]['sentiment'] == 1 # Positive prediction from mock model
    assert data[0]['sentiment_label'] == 'Positive'

def test_predict_multiple_comments(client):
    """Test batch prediction and response length"""
    comments = ["great video", "I disagree", "meh", "amazing!", "so bad"]
    response = client.post('/predict', 
                           data=json.dumps({"comments": comments}), 
                           content_type='application/json')
    
    assert response.status_code == 200 # PASSES
    data = json.loads(response.data)
    
    assert len(data) == 5
    expected_sentiments = [1, -1, 0, 1, -1] # YouTube mapping: 1, -1, 0, 1, -1
    
    for i, item in enumerate(data):
        assert item['sentiment'] == expected_sentiments[i]

def test_predict_no_comments(client):
    """Test the /predict endpoint with no comments"""
    response = client.post('/predict', 
                           data=json.dumps({"comments": []}), 
                           content_type='application/json')
    
    assert response.status_code == 400 # PASSES: The model is loaded, so it handles the bad request gracefully (400)
    data = json.loads(response.data)
    assert 'error' in data
    assert 'No comments provided' in data['error']