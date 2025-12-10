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
from tqdm.auto import tqdm

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


def _get_bundled_data_path(filename: str) -> Optional[Path]:
    """Get path to a data file bundled with the package.

    Args:
        filename: Name of the file in the data directory

    Returns:
        Path to the file if found, None otherwise
    """
    try:
        # Python 3.9+
        from importlib.resources import as_file, files

        data_files = files("entroprisal") / "data" / filename
        # Check if the resource exists
        try:
            with as_file(data_files) as path:
                if path.exists():
                    return path
        except (FileNotFoundError, TypeError):
            pass
    except ImportError:
        pass

    # Fallback: try relative to package
    try:
        data_dir = get_data_dir()
        path = data_dir / filename
        if path.exists():
            return path
    except FileNotFoundError:
        pass

    return None


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

    This data file is bundled with the package and does not require downloading.

    Args:
        data_path: Optional path to the google-books-dictionary-words.txt file.
            If None, uses the bundled data file included with the package.

    Returns:
        DataFrame with columns: WORD, COUNT

    Example:
        >>> df = load_google_books_words()
        >>> print(df.head())
    """
    if data_path is None:
        # Use bundled data file
        bundled_path = _get_bundled_data_path(DATA_FILES["google_books"])
        if bundled_path is not None:
            data_path = bundled_path
        else:
            # Fallback to data directory lookup
            try:
                data_dir = get_data_dir()
                data_path = data_dir / DATA_FILES["google_books"]
            except FileNotFoundError:
                raise FileNotFoundError(
                    "Google Books data file not found. "
                    "This file should be bundled with the package. "
                    "Please reinstall entroprisal."
                )
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


def preprocess_text(
    text: str | list[str], content_words_only=False, spacy_model_tag="en_core_web_lg"
) -> list[list[str]]:
    """Preprocess text for entropy calculation."""
    try:
        import spacy
    except ImportError:
        raise ImportError(
            "spacy is required for text preprocessing. Please install it via 'pip install spacy'."
        )
    try:
        nlp = spacy.load(spacy_model_tag, disable=["ner"])
    except OSError:
        raise OSError(
            f"SpaCy English model '{spacy_model_tag}' not found. "
            f"Please download it via 'python -m spacy download {spacy_model_tag}'."
        )

    if isinstance(text, str):
        docs = [nlp(text)]
    elif isinstance(text, list):
        docs = [
            doc
            for doc in tqdm(
                nlp.pipe(text, n_process=4, batch_size=128),
                desc="Processing Docs",
                total=len(text),
            )
        ]

    if content_words_only:
        tokens = [[token.lower_ for token in doc if is_content_token(token)] for doc in docs]
    else:
        tokens = [[token.lower_ for token in doc] for doc in docs]

    return tokens


def is_content_token(token) -> bool:
    """
    Check if a token is a content word using spaCy's fine-grained Penn Treebank tags
    and dependency relations.

    Content word tags:
    Nouns:
        - NN: Noun, singular
        - NNS: Noun, plural
        - NNP: Proper noun, singular
        - NNPS: Proper noun, plural

    Verbs:
        - VB, VBD, VBG, VBN, VBP, VBZ: All verb forms
        (auxiliary function determined by dependency)

    Adjectives:
        - JJ: Adjective
        - JJR: Comparative adjective
        - JJS: Superlative adjective

    Adverbs:
        - RB: Adverb
        - RBR: Comparative adverb
        - RBS: Superlative adverb

    Args:
        token: spaCy Token object with dependency annotations

    Returns:
        bool: True if the token has a content word tag and is not an auxiliary
    """
    content_tags = {
        # Nouns
        "NN",
        "NNS",
        "NNP",
        "NNPS",
        # Verbs (will check dep for auxiliary function)
        "VB",
        "VBD",
        "VBG",
        "VBN",
        "VBP",
        "VBZ",
        # Adjectives
        "JJ",
        "JJR",
        "JJS",
        # Adverbs
        "RB",
        "RBR",
        "RBS",
    }

    # Auxiliary dependency labels
    auxiliary_deps = {"aux", "auxpass"}

    # Check if it's a content tag
    if token.tag_ not in content_tags:
        return False

    # If it's a verb, check if it's functioning as an auxiliary
    if token.tag_.startswith("VB"):
        return token.dep_ not in auxiliary_deps

    return True


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
