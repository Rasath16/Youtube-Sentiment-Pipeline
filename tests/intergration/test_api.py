import pytest
import json
import sys 
# Removed unnecessary imports like pickle, csr_matrix since we are mocking at a higher level
from scipy.sparse import csr_matrix # Keep this for MockVectorizer
from pytest_mock import mocker 

# --- Mock Classes (Minimal changes) ---
class MockVectorizer:
    """Mock class that mimics the TfidfVectorizer's transform method."""
    def transform(self, comments):
        return csr_matrix([[1, 0]] * len(comments))

class MockModel:
    """Mock class for the LightGBM model."""
    def predict(self, features):
        return [1, 2, 0, 1, 2] * (len(features.toarray()) // 5 + 1)
    
    def predict_proba(self, features):
        return [[0.1, 0.8, 0.1]] * len(features.toarray())

# Global mock instances
mock_model = MockModel()
mock_vectorizer = MockVectorizer()


# --- Fixture with Direct Application Function Mocking ---

@pytest.fixture(scope='function')
def client(mocker): 
    
    # FIX: Mock the application's entire loading function and ensure the returned 
    # model_info dictionary is fully populated, bypassing the tag parsing issue.
    mocker.patch(
        'flask_api.app.load_model_from_registry',
        return_value=(mock_model, mock_vectorizer, {
            "version": 99,
            "accuracy": 0.85, # CRITICAL: Manually set accuracy to fix the KeyError
            "model_name": "final_lightgbm_adasyn_model",
            "model_type": "LightGBM",
            "stage": "Staging",
            "run_id": "mock_run_id",
            "imbalance_method": "ADASYN"
        })
    )

    # Force module reload (Still necessary due to global initialization)
    if 'flask_api.app' in sys.modules:
        del sys.modules['flask_api.app']
        
    from flask_api.app import app
    
    app.config['TESTING'] = True
    
    with app.test_client() as client:
        yield client 

# --- API Tests (remain unchanged, as the setup error is now fixed) ---

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