import pytest
import pandas as pd
from src.data.data_preprocessing import preprocess_comment, normalize_text

# Mock parameters needed by normalize_text
@pytest.fixture
def mock_params():
    return {
        'text_preprocessing': {
            'remove_stopwords': True,
            'lowercase': True,
            'remove_punctuation': False,
            'remove_numbers': False,
            'lemmatization': True
        }
    }



def test_preprocess_comment_lowercase():
    # ... (test remains unchanged)
    comment = "This Is A TEST Comment!"
    expected = "test comment!"
    assert preprocess_comment(comment, lowercase=True) == expected

def test_preprocess_comment_stopwords():
    # ... (test remains unchanged)
    comment = "This is a very good product but I did not like the packaging."
    expected = "good product but not like packaging."
    assert preprocess_comment(comment) == expected

def test_preprocess_comment_punctuation_removal():
    # Test full punctuation removal
    comment = "Hello, world! What's up? (123)"
    expected = "hello world whats 123" 
    assert preprocess_comment(comment, remove_punctuation=True) == expected

def test_preprocess_comment_lemmatization():
    # Test lemmatization
    comment = "The dogs were running happily."
    expected = "dog running happily."
    assert preprocess_comment(comment) == expected

def test_preprocess_comment_mixed_case():
    # ... (test remains unchanged)
    comment = "  Nice\nVid  Too much    space. "
    expected = "nice vid much space."
    assert preprocess_comment(comment) == expected


@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({
        'clean_comment': [
            "This is amazing!",
            "I hate this.", # Category -1 -> 2
            "Neutral comment, nothing to see.",
            "  This comment becomes empty after processing. ", # Expected to remain 'comment'
            "Great work, 10/10!"
        ],
        'category': [1, -1, 0, 1, 1] 
    })

def test_normalize_text_label_remapping(sample_dataframe, mock_params):
    # Test label remapping: -1 -> 2, 0 -> 0, 1 -> 1
    df_processed = normalize_text(sample_dataframe.copy(), mock_params)
    
    assert list(df_processed['category'].unique()) == [1, 2, 0]
    assert list(df_processed[df_processed['clean_comment'] == 'hate this.']['category']) == [2]

def test_normalize_text_empty_row_removal(sample_dataframe, mock_params):
    # Test removal of rows that become empty after cleaning
    params_remove_all = {
        'text_preprocessing': {
            'remove_stopwords': True,
            'lowercase': True,
            'remove_punctuation': True,
            'remove_numbers': True,
            'lemmatization': True
        }
    }
    
    # The comment "  This comment becomes empty after processing. " actually becomes 'comment' with remove_all=True
    df_processed = normalize_text(sample_dataframe.copy(), params_remove_all)
    
    assert len(df_processed) == 5
    assert 'This comment becomes empty after processing.' not in df_processed['clean_comment'].tolist()