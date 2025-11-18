import pytest
import json
import sys # <-- CRITICAL IMPORT
from scipy.sparse import csr_matrix 
# from pytest_mock import mocker # No longer needed if not using mocker.patch

# --- Mock Classes ---
class MockVectorizer:
    """Mock class that mimics the TfidfVectorizer's transform method."""
    def transform(self, comments):
        # Always return a placeholder sparse matrix
        return csr_matrix([[1, 0]] * len(comments))

class MockModel:
    """Mock class for the LightGBM model."""
    def predict(self, features):
        # Internal mapping: 0: Neutral, 1: Positive, 2: Negative
        return [1, 2, 0, 1, 2] * (len(features.toarray()) // 5 + 1)
    
    def predict_proba(self, features):
        # Simulate high confidence prediction
        return [[0.1, 0.8, 0.1]] * len(features.toarray())

# --- Fixture with Direct Variable Injection on Module ---

@pytest.fixture(scope='function')
def client(): 
    
    # 1. Force reload if necessary (ensures a clean test state for globals)
    if 'flask_api.app' in sys.modules:
        del sys.modules['flask_api.app']
        
    # 2. Import the app. This runs the global initialization in app.py, 
    # which will FAIL and set the global variables to None and {}.
    from flask_api.app import app
    
    # 3. CRITICAL FIX: Use sys.modules to access the module's namespace and 
    # manually override the failed global variables with mock instances.
    module = sys.modules['flask_api.app']
    
    module.model = MockModel()
    module.vectorizer = MockVectorizer()
    module.model_info = {
        "version": 99,
        "accuracy": 0.85, # Manually set for test_home_endpoint
        "model_name": "final_lightgbm_adasyn_model",
        "model_type": "LightGBM",
        "stage": "Staging",
        "run_id": "mock_run_id",
        "imbalance_method": "ADASYN"
    }

    # 4. Set testing configuration and yield client
    app.config['TESTING'] = True
    
    with app.test_client() as client:
        yield client 

# --- API Tests (remain unchanged) ---

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

def test_predict_single_comment(client):
    """Test the /predict endpoint with a single comment"""
    comment = "I loved this video, you are the best!"
    response = client.post('/predict', 
                           data=json.dumps({"comments": [comment]}), 
                           content_type='application/json')
    
    assert response.status_code == 200 
    data = json.loads(response.data)
    
    assert len(data) == 1
    assert data[0]['sentiment'] == 1 
    assert data[0]['sentiment_label'] == 'Positive'

def test_predict_multiple_comments(client):
    """Test batch prediction and response length"""
    comments = ["great video", "I disagree", "meh", "amazing!", "so bad"]
    response = client.post('/predict', 
                           data=json.dumps({"comments": comments}), 
                           content_type='application/json')
    
    assert response.status_code == 200
    data = json.loads(response.data)
    
    assert len(data) == 5
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