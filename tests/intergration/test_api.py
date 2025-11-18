import pytest
import json
from unittest.mock import MagicMock, patch

import pytest
import json
# from unittest.mock import MagicMock, patch # REMOVED: No longer needed

# Note: We need to import the app only when necessary to control imports
# We use a fixture to configure the app for testing
from scipy.sparse import csr_matrix # Added missing import

# A simple mock model and vectorizer for local API testing
class MockVectorizer:
    def transform(self, comments):
        # Always return a placeholder sparse matrix
        # Use csr_matrix from scipy.sparse
        return csr_matrix([[1, 0]] * len(comments))

class MockModel:
    def predict(self, features):
        # Predicts [1, 2, 0, 1, ...] (Positive, Negative, Neutral, Positive, ...)
        # The internal mapping is 0: Neutral, 1: Positive, 2: Negative
        return [1, 2, 0, 1, 2] * (len(features.toarray()) // 5 + 1)
    
    def predict_proba(self, features):
        # Simulate high confidence prediction
        return [[0.1, 0.8, 0.1]] * len(features.toarray())

mock_model = MockModel()
mock_vectorizer = MockVectorizer()
mock_model_info = {
    "version": 99,
    "accuracy": 0.85
}

# Fix: Use 'mocker' fixture from pytest-mock instead of @patch decorator
@pytest.fixture(scope='module')
def client(mocker): # FIX 1: Change 'mock_loader' to 'mocker'
    # Setup: Mock the model loading function using mocker
    mocker.patch( # FIX 2: Use mocker.patch
        'flask_api.app.load_model_from_registry',
        return_value=(mock_model, mock_vectorizer, mock_model_info)
    )
    
    # Import the app after patching to ensure the global model loading uses the mock
    from flask_api.app import app
    app.config['TESTING'] = True
    
    # Use Flask's built-in test client
    with app.test_client() as client:
        yield client 

# --- API Health Check Tests ---

def test_health_check(client):
    """Test the /health endpoint"""
    response = client.get('/health')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['status'] == 'healthy'
    assert data['model_loaded'] == True

def test_home_endpoint(client):
    """Test the root endpoint"""
    response = client.get('/')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['status'] == 'running'
    assert data['model_accuracy'] == '85.00%'

# --- API Prediction Tests ---

def test_predict_single_comment(client):
    """Test the /predict endpoint with a single comment"""
    comment = "I loved this video, you are the best!"
    response = client.post('/predict', 
                           data=json.dumps({"comments": [comment]}), 
                           content_type='application/json')
    
    assert response.status_code == 200
    data = json.loads(response.data)
    
    # Expected output (from MockModel prediction=1) is Positive (YouTube: 1)
    assert len(data) == 1
    assert data[0]['comment'] == comment
    assert data[0]['sentiment'] == 1
    assert data[0]['sentiment_label'] == 'Positive'
    assert data[0]['confidence'] > 0.7 

def test_predict_multiple_comments(client):
    """Test batch prediction and response length"""
    comments = ["great video", "I disagree", "meh", "amazing!", "so bad"]
    response = client.post('/predict', 
                           data=json.dumps({"comments": comments}), 
                           content_type='application/json')
    
    assert response.status_code == 200
    data = json.loads(response.data)
    
    assert len(data) == 5
    # Check predictions: 1 (Pos), 2 (Neg), 0 (Neu), 1 (Pos), 2 (Neg)
    # YouTube mapping: 1, -1, 0, 1, -1
    expected_sentiments = [1, -1, 0, 1, -1]
    
    for i, item in enumerate(data):
        assert item['sentiment'] == expected_sentiments[i]

def test_predict_no_comments(client):
    """Test the /predict endpoint with no comments"""
    response = client.post('/predict', 
                           data=json.dumps({"comments": []}), 
                           content_type='application/json')
    
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data
    assert 'No comments provided' in data['error']