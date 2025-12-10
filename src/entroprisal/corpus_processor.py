"""Corpus processing utilities using SpaCy.

This module provides functions for processing text corpora using SpaCy,
creating DocBin files for efficient storage and retrieval of processed documents.

Note: This module requires SpaCy to be installed separately:
    pip install spacy
    python -m spacy download en_core_web_lg
"""

import logging
from pathlib import Path
from typing import Iterator, Optional, Tuple

import pandas as pd
import spacy
from spacy.tokens import DocBin
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


def generate_text_metadata_pairs(
    df: pd.DataFrame, text_col: str, metadata_col: str
) -> Iterator[Tuple[str, str]]:
    """Generate pairs of (text, metadata) from dataframe.

    Args:
        df: Input DataFrame
        text_col: Name of column containing text to process
        metadata_col: Name of column containing metadata to preserve

    Yields:
        Tuples of (text, metadata) for each row

    Note:
        Assumes column names are valid Python attributes (no spaces).
    """
    for row in df.itertuples():
        yield (getattr(row, text_col), getattr(row, metadata_col))


def process_dataframe(
    df: pd.DataFrame,
    text_col: str,
    metadata_col: str,
    output_dir: str | Path,
    model: str = "en_core_web_lg",
    n_process: int = 32,
    batch_size: int = 512,
    save_interval: int = 10_000,
):
    """Process dataframe using parallel processing and save to multiple DocBin files.

    Args:
        df: Input DataFrame containing texts and metadata
        text_col: Name of column containing text to process
        metadata_col: Name of column containing metadata to preserve
        output_dir: Directory to save the DocBin files
        model: Name of SpaCy model to use (default: "en_core_web_lg")
        n_process: Number of processes for parallel processing (default: 32)
        batch_size: Documents to process in each batch (default: 512)
        save_interval: Save DocBin every N documents (default: 10,000)

    Returns:
        None. Saves processed documents to {output_dir}/processed_docs_*.spacy

    Example:
        >>> df = pd.DataFrame({
        ...     "text": ["Sample text 1", "Sample text 2"],
        ...     "metadata": ["meta1", "meta2"]
        ... })
        >>> process_dataframe(
        ...     df=df,
        ...     text_col="text",
        ...     metadata_col="metadata",
        ...     output_dir="processed_docs",
        ...     n_process=16,
        ...     batch_size=100
        ... )
    """
    nlp = spacy.load(model, disable=["ner"])
    nlp.max_length = 5_000_000
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    docs = []
    current_docbin_idx = 0

    # Process using nlp.pipe() with parallel processing and as_tuples=True
    with tqdm(total=len(df), desc="Processing documents") as pbar:
        for i, (doc, meta) in enumerate(
            nlp.pipe(
                generate_text_metadata_pairs(df, text_col, metadata_col),
                as_tuples=True,
                n_process=n_process,
                batch_size=batch_size,
            )
        ):
            # Store metadata
            doc.user_data["meta"] = meta

            # Add to current DocBin and check size
            docs.append(doc)

            # Save at specified intervals
            if i % save_interval == 0 and i > 0:
                output_path = output_dir / f"processed_docs_{current_docbin_idx:02}.spacy"
                docbin = DocBin(docs=docs, store_user_data=True)
                docbin.to_disk(output_path)
                logger.info(f"Saved DocBin {current_docbin_idx} to {output_path}")
                current_docbin_idx += 1
                docs = []

            # Update progress bar
            pbar.update(1)

    # Save final DocBin if it contains any documents
    if len(docs) > 0:
        output_path = output_dir / f"processed_docs_{current_docbin_idx:02}.spacy"
        docbin = DocBin(docs=docs, store_user_data=True)
        docbin.to_disk(output_path)
        logger.info(f"Saved final DocBin {current_docbin_idx} to {output_path}")


def load_all_docbins(directory: str | Path, model: Optional[str] = None):
    """Generator function to load and yield all docs from multiple DocBin files.

    Args:
        directory: Directory containing .spacy DocBin files
        model: SpaCy model name to use for loading. If None, uses blank English model.

    Yields:
        SpaCy Doc objects from all DocBin files in directory

    Example:
        >>> for doc in load_all_docbins("processed_docs"):
        ...     print(doc.text)
        ...     print(doc.user_data.get("meta"))
    """
    directory = Path(directory)

    if model:
        nlp = spacy.load(model)
    else:
        nlp = spacy.blank("en")  # Light-weight model just for loading

    for file_path in sorted(directory.glob("*.spacy")):
        logger.info(f"Loading {file_path}")
        doc_bin = DocBin().from_disk(file_path)
        for doc in doc_bin.get_docs(nlp.vocab):
            yield doc
