"""Tests for CharacterEntropisalCalculator."""

import pandas as pd
import pytest

from entroprisal import CharacterEntropisalCalculator


@pytest.fixture
def sample_words():
    """Create sample word frequency data for testing."""
    return pd.DataFrame(
        {
            "WORD": ["cat", "dog", "the", "and", "can", "car"],
            "COUNT": [1000, 800, 5000, 4000, 600, 700],
        }
    )


def test_init(sample_words):
    """Test calculator initialization."""
    calc = CharacterEntropisalCalculator(sample_words)
    assert calc is not None
    assert len(calc.df) == 6


def test_calculate_metrics(sample_words):
    """Test basic metric calculation."""
    calc = CharacterEntropisalCalculator(sample_words)
    tokens = ["the", "cat", "and", "dog"]
    metrics = calc.calculate_metrics(tokens)

    assert "char_entropy" in metrics
    assert "char_surprisal" in metrics
    assert "bigraph_entropy" in metrics
    assert "bigraph_surprisal" in metrics
    assert "trigraph_entropy" in metrics
    assert "trigraph_surprisal" in metrics


def test_calculate_batch(sample_words):
    """Test batch processing."""
    calc = CharacterEntropisalCalculator(sample_words)
    token_lists = [["the", "cat"], ["the", "dog"]]

    results = calc.calculate_batch(token_lists)
    assert len(results) == 2


def test_get_character_entropy(sample_words):
    """Test character entropy lookup."""
    calc = CharacterEntropisalCalculator(sample_words)

    # Should return a value for common character
    entropy = calc.get_character_entropy("c")
    assert entropy is not None
    assert entropy >= 0


def test_get_character_surprisal(sample_words):
    """Test character surprisal lookup."""
    calc = CharacterEntropisalCalculator(sample_words)

    # Test surprisal lookup
    surprisal = calc.get_character_surprisal("c", "a")
    assert surprisal is None or surprisal >= 0
