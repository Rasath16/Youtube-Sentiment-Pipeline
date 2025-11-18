import pytest
import json
import sys 
import pickle 
from scipy.sparse import csr_matrix 
from pytest_mock import mocker 

# --- Mock Classes (remain unchanged) ---
class MockVectorizer:
    def transform(self, comments):
        return csr_matrix([[1, 0]] * len(comments))

class MockModel:
    def predict(self, features):
        return [1, 2, 0, 1, 2] * (len(features.toarray()) // 5 + 1)
    
    def predict_proba(self, features):
        return [[0.1, 0.8, 0.1]] * len(features.toarray())

mock_model = MockModel()
mock_vectorizer = MockVectorizer()
mock_model_info = {
    # ... (remains unchanged)
}

# --- Fixture with Dependency Mocking ---

@pytest.fixture(scope='function')
def client(mocker, tmp_path): 
    
    # FIX: Create a list of mock tag objects with key and value attributes
    mock_tags = [
        mocker.Mock(key='test_accuracy', value='0.85'),
        mocker.Mock(key='test_f1_weighted', value='0.80'),
        mocker.Mock(key='f1_positive', value='0.90'), # Added for completeness as per app.py
        mocker.Mock(key='f1_neutral', value='0.70'),
        mocker.Mock(key='f1_negative', value='0.75'),
    ]

    # 1. Mock MlflowClient and its methods
    mock_client = mocker.Mock()
    # Ensure get_latest_versions returns a mock object with required properties
    mock_client.get_latest_versions.return_value = [
        mocker.Mock(
            version=99, 
            run_id='mock_run_id', 
            current_stage='Staging', 
            tags=mock_tags # <-- CORRECTED: Passing the list of mock tag objects
        )
    ]
    mocker.patch('flask_api.app.MlflowClient', return_value=mock_client)

    # 2. Mock model loading
    mocker.patch('flask_api.app.mlflow.sklearn.load_model', return_value=mock_model)
    
    # 3. Mock vectorizer downloading
    vectorizer_path = tmp_path / 'tfidf_vectorizer.pkl'
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(mock_vectorizer, f)
    mocker.patch('flask_api.app.mlflow.artifacts.download_artifacts', return_value=str(vectorizer_path))

    # 4. Force module reload (Critical for making global code run with mocks)
    if 'flask_api.app' in sys.modules:
        del sys.modules['flask_api.app']
        
    from flask_api.app import app
    
    app.config['TESTING'] = True
    
    with app.test_client() as client:
        yield client 

# --- API Tests (remain unchanged, as the error was in the setup) ---

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