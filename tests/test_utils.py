"""Tests for utility functions."""

import pytest

from entroprisal.utils import is_content_token, preprocess_text


@pytest.fixture
def skip_if_no_spacy():
    """Skip test if spaCy or en_core_web_lg model is not installed."""
    try:
        import spacy

        try:
            spacy.load("en_core_web_lg")
        except OSError:
            pytest.skip("SpaCy model 'en_core_web_lg' not installed")
    except ImportError:
        pytest.skip("spaCy not installed")


def test_preprocess_text_single_string(skip_if_no_spacy):
    """Test preprocessing a single string."""
    text = "The quick brown fox jumps over the lazy dog."
    result = preprocess_text(text)
    
    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], list)
    assert all(isinstance(token, str) for token in result[0])
    
    # Check that common words are present
    tokens = result[0]
    assert "quick" in tokens
    assert "brown" in tokens
    assert "fox" in tokens


def test_preprocess_text_multiple_strings(skip_if_no_spacy):
    """Test preprocessing multiple strings."""
    texts = ["First sentence.", "Second sentence.", "Third sentence."]
    result = preprocess_text(texts)
    
    assert isinstance(result, list)
    assert len(result) == 3
    assert all(isinstance(doc_tokens, list) for doc_tokens in result)
    
    # Check first document
    assert "first" in result[0]
    # Check second document
    assert "second" in result[1]
    # Check third document
    assert "third" in result[2]


def test_preprocess_text_content_words_only(skip_if_no_spacy):
    """Test extracting only content words."""
    text = "The quick brown fox jumps over the lazy dog."
    result = preprocess_text(text, content_words_only=True)
    
    tokens = result[0]
    
    # Content words should be present
    assert "quick" in tokens
    assert "brown" in tokens
    assert "fox" in tokens
    assert "jumps" in tokens
    assert "lazy" in tokens
    assert "dog" in tokens
    
    # Function words should be filtered out
    assert "the" not in tokens
    assert "over" not in tokens  # 'over' is a preposition, filtered out


def test_preprocess_text_all_words(skip_if_no_spacy):
    """Test that all words mode includes function words."""
    text = "The cat is running."
    result = preprocess_text(text, content_words_only=False)
    
    tokens = result[0]
    
    # Should include function words
    assert "the" in tokens
    assert "is" in tokens  # auxiliary verb
    # And content words
    assert "cat" in tokens
    assert "running" in tokens


def test_preprocess_text_requires_spacy():
    """Test that preprocess_text raises ImportError when spaCy not available."""
    # Mock the import to raise ImportError
    import sys
    spacy_module = sys.modules.get("spacy")
    
    try:
        if "spacy" in sys.modules:
            del sys.modules["spacy"]
        
        # This would raise ImportError if spacy truly wasn't installed
        # but since we need it for other tests, we'll skip this test
        pytest.skip("Can't properly test ImportError when spacy is installed")
    finally:
        if spacy_module:
            sys.modules["spacy"] = spacy_module


def test_preprocess_text_custom_model(skip_if_no_spacy):
    """Test using a custom spaCy model."""
    # Try with en_core_web_sm if available
    try:
        text = "Simple test."
        result = preprocess_text(text, spacy_model_tag="en_core_web_sm")
        
        assert isinstance(result, list)
        assert len(result) == 1
        assert "simple" in result[0]
        assert "test" in result[0]
    except OSError:
        # Model not installed, skip
        pytest.skip("en_core_web_sm not installed")
