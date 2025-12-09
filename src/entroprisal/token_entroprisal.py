"""Token-level entropy and surprisal calculator.

This module calculates entropy and surprisal metrics based on n-gram token frequencies
from reference corpora. It uses pre-computed transition matrices for efficient batch processing.
"""

import logging
from pathlib import Path
from typing import Dict, List, Union, cast

import numpy as np
import pandas as pd
import polars as pl

logger = logging.getLogger(__name__)


class TokenEntropisalCalculator:
    """Calculate token-level entropy and surprisal metrics.

    This calculator uses n-gram token frequencies to compute:
    - Surprisal: Information content of each token given context
    - Entropy: Uncertainty about the next token given context

    Attributes:
        EOW: End-of-word marker
        min_frequency: Minimum frequency threshold for n-grams
    """

    EOW = "_"

    def __init__(
        self,
        ngram_frequencies: Union[pl.LazyFrame, pl.DataFrame, Path, str],
        min_frequency: int = 100,
    ):
        """Initialize the calculator with n-gram frequency data.

        Args:
            ngram_frequencies: LazyFrame, DataFrame, or path to parquet file containing
                columns: token_0, token_1, token_2, token_3, count
            min_frequency: Minimum frequency threshold to include n-grams (default: 100)
        """
        self.min_frequency = min_frequency

        # Load data if path provided
        if isinstance(ngram_frequencies, (str, Path)):
            ngram_frequencies = pl.scan_parquet(str(ngram_frequencies))
        elif isinstance(ngram_frequencies, pl.DataFrame):
            ngram_frequencies = ngram_frequencies.lazy()

        # Pre-collect filtered data
        self.ngram_df = ngram_frequencies.filter(pl.col("count") >= self.min_frequency).collect()

        # Pre-compute transition matrices once
        self._build_all_token_transitions()
        self._build_surprisal_lookup()
        self._build_entropy_lookup()

    def _build_all_token_transitions(self):
        """Pre-compute all token transition matrices."""
        self._token_transitions = {}
        for n in range(1, 4):
            context_cols = [f"token_{i}" for i in range(3 - n, 3)]

            self._token_transitions[n] = (
                self.ngram_df.select([*context_cols, "token_3", "count"])
                .filter(~pl.any_horizontal([pl.col(col) == "#" for col in context_cols]))
                .group_by([*context_cols, "token_3"])  # Group by all token columns
                .agg(pl.col("count").sum())  # Sum counts for duplicates
                .with_columns(
                    [
                        # Create a composite context key for easier joining
                        pl.concat_str(context_cols, separator="|||").alias("context_key")
                    ]
                )
            )

    def _build_surprisal_lookup(self):
        """Build surprisal lookup dataframes."""
        self._surprisal_lookup = {}
        for n in range(1, 4):
            # Calculate context totals
            context_totals = (
                self._token_transitions[n]
                .group_by("context_key")
                .agg(pl.col("count").sum().alias("context_total"))
            )

            # Calculate surprisal for each context-target pair
            self._surprisal_lookup[n] = (
                self._token_transitions[n]
                .join(context_totals, on="context_key")
                .with_columns(
                    [(-(pl.col("count") / pl.col("context_total")).log(2)).alias("surprisal")]
                )
                .select(["context_key", "token_3", "surprisal"])
                .rename({"token_3": "target"})
            )

    def _build_entropy_lookup(self):
        """Build entropy lookup dataframes."""
        self._entropy_lookup = {}
        for n in range(1, 4):
            # Calculate context totals first
            context_totals = (
                self._token_transitions[n]
                .group_by("context_key")
                .agg(pl.col("count").sum().alias("context_total"))
            )

            # Join with context totals and calculate probabilities and entropy
            self._entropy_lookup[n] = (
                self._token_transitions[n]
                .join(context_totals, on="context_key")
                .with_columns([(pl.col("count") / pl.col("context_total")).alias("prob")])
                .group_by("context_key")
                .agg([(-(pl.col("prob") * pl.col("prob").log(2)).sum()).alias("entropy")])
            )

    def _extract_ngrams(self, tokens: List[str], n: int) -> pl.DataFrame:
        """Extract all n-grams from tokens.

        Args:
            tokens: List of token strings
            n: N-gram order (1, 2, or 3)

        Returns:
            DataFrame with columns: context_key, target, position
        """
        # Create all possible n-grams
        ngrams = []
        for i in range(len(tokens) - n):
            context = tokens[i : i + n]
            target = tokens[i + n]
            context_key = "|||".join(context)
            ngrams.append({"context_key": context_key, "target": target, "position": i})

        return pl.DataFrame(ngrams)

    def _calculate_ngram_metrics(self, tokens: List[str], n: int) -> Dict[str, float]:
        """Calculate n-gram metrics using efficient merge operations.

        Args:
            tokens: List of token strings
            n: N-gram order (1, 2, or 3)

        Returns:
            Dictionary of metric names and values
        """
        # Extract all n-grams from input tokens
        input_ngrams = self._extract_ngrams(tokens, n)

        if len(input_ngrams) == 0:
            return {}

        metrics = {}

        # Calculate surprisal
        surprisals = input_ngrams.join(
            self._surprisal_lookup[n], on=["context_key", "target"], how="inner"
        )

        # Mean surprisal
        metrics[f"ngram_surprisal_{n}_support"] = len(surprisals)
        if len(surprisals) > 0:
            mean_surprisal = cast(float, surprisals["surprisal"].mean())
            metrics[f"ngram_surprisal_{n}"] = mean_surprisal

        # Calculate entropy
        entropies = input_ngrams.join(self._entropy_lookup[n], on="context_key", how="inner")

        # Mean entropy
        metrics[f"ngram_entropy_{n}_support"] = len(entropies)
        if len(entropies) > 0:
            mean_entropy = cast(float, entropies["entropy"].mean())
            metrics[f"ngram_entropy_{n}"] = mean_entropy

        return metrics

    def calculate_metrics(self, tokens: List[str]) -> Dict[str, float]:
        """Calculate all entropy and surprisal metrics for a token sequence.

        Args:
            tokens: List of token strings

        Returns:
            Dictionary mapping metric names to values. Includes:
                - n_tokens: Number of tokens
                - mean_token_length: Average token length in characters
                - ngram_surprisal_{1,2,3}: Mean surprisal for each n-gram order
                - ngram_surprisal_{1,2,3}_support: Number of n-grams with coverage
                - ngram_entropy_{1,2,3}: Mean entropy for each n-gram order
                - ngram_entropy_{1,2,3}_support: Number of contexts with coverage
        """
        metrics = {}

        # Basic metrics
        metrics["n_tokens"] = len(tokens)
        metrics["mean_token_length"] = float(np.mean([len(tok) for tok in tokens]))

        # Calculate metrics for each n-gram size
        for n in range(1, 4):
            # N-gram metrics
            ngram_metrics = self._calculate_ngram_metrics(tokens, n)
            metrics.update(ngram_metrics)

        return {k: v for k, v in metrics.items() if not np.isnan(v)}

    def calculate_batch(self, token_lists: List[List[str]]) -> pd.DataFrame:
        """Calculate metrics for multiple token sequences.

        Args:
            token_lists: List of token lists

        Returns:
            DataFrame with one row per input sequence and columns for each metric
        """
        results = [self.calculate_metrics(tokens) for tokens in token_lists]
        return pd.DataFrame(results)

    def get_detailed_ngram_analysis(self, tokens: List[str]) -> Dict[int, pd.DataFrame]:
        """Get detailed analysis of individual n-grams with their entropies and surprisals.

        Args:
            tokens: List of token strings

        Returns:
            Dictionary mapping n-gram order to DataFrames with columns:
                - context_key: The context tokens joined
                - target: The target token
                - position: Position in the token sequence
                - surprisal: Surprisal value
                - entropy: Entropy value
                - ngram_order: N-gram order (1, 2, or 3)
        """
        detailed_results = {}

        for n in range(1, 4):
            # Extract all n-grams from input tokens
            input_ngrams = self._extract_ngrams(tokens, n)

            if len(input_ngrams) == 0:
                detailed_results[n] = pd.DataFrame()
                continue

            # Inner join with surprisal lookup
            result_df = input_ngrams.join(
                self._surprisal_lookup[n], on=["context_key", "target"], how="inner"
            )

            # Inner join with entropy lookup
            result_df = result_df.join(self._entropy_lookup[n], on="context_key", how="inner")

            # Convert to pandas and add n-gram order
            result_df = result_df.to_pandas().set_index("position").sort_index()
            result_df["ngram_order"] = n
            detailed_results[n] = result_df

        return detailed_results
