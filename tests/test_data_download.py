"""Tests for data download functionality."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from entroprisal.utils import (
    DATA_FILES,
    DIRECT_URLS,
    HF_REPO_ID,
    _direct_download,
    _download_from_hf,
    ensure_data_file,
)


def test_ensure_data_file_uses_local_first(tmp_path):
    """Test that ensure_data_file uses local files when available."""
    # Create a mock local file
    local_file = tmp_path / "test.txt"
    local_file.write_text("test data")
    
    # Mock get_data_dir to return our tmp_path
    with patch("entroprisal.utils.get_data_dir", return_value=tmp_path):
        result = ensure_data_file("test.txt")
        assert result == local_file
        assert result.exists()


def test_constants_are_defined():
    """Test that download configuration constants are properly defined."""
    assert HF_REPO_ID == "langdonholmes/slimpajama-ngrams"
    assert "4grams_aw" in DATA_FILES
    assert "4grams_cw" in DATA_FILES
    assert "google_books" in DATA_FILES
    assert "4grams_aw" in DIRECT_URLS
    assert "4grams_cw" in DIRECT_URLS


def test_hf_download_called_when_available():
    """Test that HF Hub download is attempted when huggingface-hub is available."""
    mock_path = Path("/fake/cache/file.parquet")
    
    with patch("entroprisal.utils.get_data_dir", side_effect=FileNotFoundError):
        with patch("huggingface_hub.hf_hub_download", return_value=str(mock_path)) as mock_hf:
            result = ensure_data_file("test.parquet")
            assert result == mock_path
            mock_hf.assert_called_once()


def test_direct_download_fallback():
    """Test that direct download is used when HF Hub fails."""
    import entroprisal.utils
    
    mock_direct_path = Path("/fake/download/4grams_aw.parquet")
    
    with patch("entroprisal.utils.get_data_dir", side_effect=FileNotFoundError):
        with patch("entroprisal.utils._download_from_hf", side_effect=Exception("HF download failed")):
            with patch.object(entroprisal.utils, "_direct_download", return_value=mock_direct_path) as mock_direct:
                result = ensure_data_file("4grams_aw.parquet")
                
                # Should fall back to direct download
                mock_direct.assert_called_once_with(
                    "4grams_aw.parquet",
                    DIRECT_URLS["4grams_aw"],
                    None
                )
                assert result == mock_direct_path


def test_load_functions_use_ensure_data_file():
    """Test that load functions properly integrate with ensure_data_file."""
    from entroprisal.utils import load_4grams, load_google_books_words

    # These should work with local files in development
    # We just verify they don't crash when data exists
    try:
        from entroprisal.utils import get_data_dir
        data_dir = get_data_dir()
        
        # Only run if we have local data
        if (data_dir / "google-books-dictionary-words.txt").exists():
            df = load_google_books_words()
            assert len(df) > 0
            assert "WORD" in df.columns
            assert "COUNT" in df.columns
        
        if (data_dir / "4grams_aw.parquet").exists():
            ngrams = load_4grams("aw")
            assert ngrams is not None
            
    except FileNotFoundError:
        # No local data, skip this test
        pytest.skip("No local data directory available")
