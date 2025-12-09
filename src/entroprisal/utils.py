"""Utility functions for the entroprisal package.

This module provides helper functions for loading reference data,
finding package resources, and common preprocessing operations.
"""

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import polars as pl

logger = logging.getLogger(__name__)

# Hugging Face Hub configuration
HF_REPO_ID = "langdonholmes/slimpajama-ngrams"
HF_REPO_TYPE = "dataset"

# Data file mappings
DATA_FILES = {
    "4grams_aw": "4grams_aw.parquet",
    "4grams_cw": "4grams_cw.parquet",
    "google_books": "google-books-dictionary-words.txt",
}

# Direct download URLs (fallback when huggingface-hub not installed)
DIRECT_URLS = {
    "4grams_aw": f"https://huggingface.co/datasets/{HF_REPO_ID}/resolve/main/4grams_aw.parquet",
    "4grams_cw": f"https://huggingface.co/datasets/{HF_REPO_ID}/resolve/main/4grams_cw.parquet",
}


def entropy(probabilities: Union[np.ndarray, list], base: float = 2.0) -> float:
    """Calculate Shannon entropy from a probability distribution.

    Args:
        probabilities: Array or list of probabilities (should sum to 1.0)
        base: Logarithm base (default: 2.0 for bits)

    Returns:
        Entropy value in units determined by base (bits if base=2)

    Example:
        >>> probs = np.array([0.5, 0.5])
        >>> entropy(probs)
        1.0
        >>> probs = np.array([0.25, 0.25, 0.25, 0.25])
        >>> entropy(probs)
        2.0
    """
    # Convert to numpy array if needed
    probs = np.asarray(probabilities, dtype=float)
    
    # Normalize to probabilities if they're counts
    if probs.sum() > 0:
        probs = probs / probs.sum()
    
    # Filter out zero probabilities to avoid log(0)
    probs = probs[probs > 0]
    
    if len(probs) == 0:
        return 0.0
    
    # H = -sum(p * log(p))
    return float(-np.sum(probs * np.log(probs) / np.log(base)))


def get_data_dir() -> Path:
    """Get the path to the data directory.

    Returns:
        Path object pointing to the data directory

    Note:
        For development, looks for data/ relative to package root.
        For installed packages, this would need to be adapted to use
        importlib.resources or downloaded data.
    """
    # Try to find data directory relative to this file
    package_root = Path(__file__).parent.parent.parent
    data_dir = package_root / "data"

    if data_dir.exists():
        return data_dir

    # Alternative: look for data in current working directory
    cwd_data = Path.cwd() / "data"
    if cwd_data.exists():
        return cwd_data

    raise FileNotFoundError(
        f"Data directory not found. Tried:\n"
        f"  - {data_dir}\n"
        f"  - {cwd_data}\n"
        f"Please ensure reference data is available."
    )


def _download_from_hf(filename: str, cache_dir: Optional[Path] = None) -> Path:
    """Download a file from Hugging Face Hub.

    Args:
        filename: Name of the file to download
        cache_dir: Optional cache directory (defaults to HF Hub default)

    Returns:
        Path to the downloaded file

    Raises:
        ImportError: If huggingface-hub is not installed
    """
    from huggingface_hub import hf_hub_download

    logger.info(f"Downloading {filename} from Hugging Face Hub...")
    
    downloaded_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=filename,
        cache_dir=str(cache_dir) if cache_dir else None,
        repo_type=HF_REPO_TYPE,
    )
    
    logger.info(f"Downloaded to {downloaded_path}")
    return Path(downloaded_path)


def _direct_download(filename: str, url: str, cache_dir: Optional[Path] = None) -> Path:
    """Download a file directly via HTTP.

    Args:
        filename: Name of the file to download
        url: Direct download URL
        cache_dir: Optional cache directory (defaults to ~/.cache/entroprisal)

    Returns:
        Path to the downloaded file
    """
    import urllib.request

    from tqdm import tqdm

    # Determine cache directory
    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "entroprisal"
    
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest_path = cache_dir / filename

    # Skip if already downloaded
    if dest_path.exists():
        logger.info(f"Using cached file: {dest_path}")
        return dest_path

    logger.info(f"Downloading {filename} from {url}...")

    # Download with progress bar
    pbar = None
    
    def reporthook(block_num, block_size, total_size):
        nonlocal pbar
        if pbar is None:
            pbar = tqdm(total=total_size, unit="B", unit_scale=True, desc=filename)
        downloaded = block_num * block_size
        if downloaded < total_size:
            pbar.update(block_size)

    try:
        urllib.request.urlretrieve(url, dest_path, reporthook=reporthook)
        if pbar is not None:
            pbar.close()
        logger.info(f"Downloaded to {dest_path}")
        return dest_path
    except Exception as e:
        if pbar is not None:
            pbar.close()
        if dest_path.exists():
            dest_path.unlink()  # Clean up partial download
        raise RuntimeError(f"Failed to download {filename}: {e}")


def ensure_data_file(filename: str, cache_dir: Optional[Path] = None) -> Path:
    """Ensure a data file is available locally, downloading if necessary.

    This function first checks if the file exists in the local data directory.
    If not found, it attempts to download from Hugging Face Hub (preferred)
    or falls back to direct HTTP download.

    Args:
        filename: Name of the file (e.g., "4grams_aw.parquet")
        cache_dir: Optional cache directory for downloads

    Returns:
        Path to the local file

    Raises:
        FileNotFoundError: If file cannot be found or downloaded
    """
    # First, try local data directory
    try:
        data_dir = get_data_dir()
        local_path = data_dir / filename
        if local_path.exists():
            logger.debug(f"Using local file: {local_path}")
            return local_path
    except FileNotFoundError:
        pass  # No local data directory, will download

    # Try Hugging Face Hub download (preferred)
    try:
        return _download_from_hf(filename, cache_dir)
    except ImportError:
        logger.info("huggingface-hub not installed, falling back to direct download")
    except Exception as e:
        logger.warning(f"Hugging Face Hub download failed: {e}, trying direct download")

    # Fallback to direct download
    # Check if we have a direct URL for this file (by filename or by key)
    direct_url = None
    if filename in DIRECT_URLS:
        direct_url = DIRECT_URLS[filename]
    else:
        # Try to find the key by matching the filename in DATA_FILES
        for key, fname in DATA_FILES.items():
            if fname == filename and key in DIRECT_URLS:
                direct_url = DIRECT_URLS[key]
                break
    
    if direct_url:
        return _direct_download(filename, direct_url, cache_dir)
    
    raise FileNotFoundError(
        f"Could not find or download {filename}. "
        f"Try installing huggingface-hub: pip install huggingface-hub"
    )


def load_google_books_words(data_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Load Google Books word frequency data.

    Args:
        data_path: Optional path to the google-books-dictionary-words.txt file.
            If None, tries to find it locally or downloads from Hugging Face Hub.

    Returns:
        DataFrame with columns: WORD, COUNT

    Example:
        >>> df = load_google_books_words()
        >>> print(df.head())
    """
    if data_path is None:
        data_path = ensure_data_file(DATA_FILES["google_books"])
    else:
        data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(f"Google Books data not found at {data_path}")

    logger.info(f"Loading Google Books word frequencies from {data_path}")
    df = pd.read_csv(data_path, sep=r"\s+", thousands=",")
    logger.info(f"Loaded {len(df):,} words")

    return df


def load_4grams(variant: str = "aw", data_path: Optional[Union[str, Path]] = None) -> pl.LazyFrame:
    """Load 4-gram frequency data.

    Args:
        variant: Either "aw" (all words) or "cw" (content words)
        data_path: Optional path to the 4grams parquet file.
            If None, tries to find it locally or downloads from Hugging Face Hub.

    Returns:
        Polars LazyFrame with n-gram frequency data

    Example:
        >>> ngrams = load_4grams("aw")
        >>> print(ngrams.head().collect())
    """
    if variant not in ["aw", "cw"]:
        raise ValueError(f"variant must be 'aw' or 'cw', got '{variant}'")

    if data_path is None:
        filename = DATA_FILES[f"4grams_{variant}"]
        data_path = ensure_data_file(filename)
    else:
        data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(f"4-gram data not found at {data_path}")

    logger.info(f"Loading 4-gram frequencies from {data_path}")
    return pl.scan_parquet(str(data_path))


def preprocess_text(text: str, aggressive: bool = False) -> str:
    """Preprocess text for entropy calculation.

    Args:
        text: Input text string
        aggressive: If True, removes all non-letter, non-space characters.
            If False, applies minimal preprocessing.

    Returns:
        Preprocessed text string

    Example:
        >>> preprocess_text("Hello, World! 123", aggressive=True)
        'hello world'
        >>> preprocess_text("Hello, World! 123", aggressive=False)
        'hello, world! 123'
    """
    import re

    if aggressive:
        # Remove all non-letter, non-space characters
        not_letter_not_space = re.compile("[^a-z ]+")
        return not_letter_not_space.sub("", text.lower())
    else:
        # Just lowercase
        return text.lower()


def get_package_version() -> str:
    """Get the version of the entroprisal package.

    Returns:
        Version string
    """
    try:
        from importlib.metadata import version

        return version("entroprisal")
    except Exception:
        return "unknown"
