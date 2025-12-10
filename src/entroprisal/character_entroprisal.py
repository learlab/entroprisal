"""Character-level entropy and surprisal calculator.

This module calculates character-level entropy and surprisal metrics based on character
transition frequencies from word frequency data. It computes both entropy and surprisal
for single characters, bigraphs, and trigraphs.
"""

import logging
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Union, cast

import numpy as np
import pandas as pd

from .utils import entropy

logger = logging.getLogger(__name__)


class CharacterEntropisalCalculator:
    """Calculate character-level entropy and surprisal metrics from word frequencies.

    This calculator builds transition frequency dictionaries from word frequency data
    and computes both entropy and surprisal metrics for characters, bigraphs, and trigraphs.

    Words are processed with boundary markers (#) at the start and end.
    For example, "cat" becomes "#cat#" for analysis.
    """

    BOUNDARY = "#"

    def __init__(
        self,
        word_frequency_data: Union[pd.DataFrame, Path, str],
        word_column: str = "WORD",
        count_column: str = "COUNT",
    ):
        """Initialize the calculator with word frequency data.

        Args:
            word_frequency_data: DataFrame or path to CSV file with word frequencies
            word_column: Name of column containing words (default: "WORD")
            count_column: Name of column containing frequency counts (default: "COUNT")
        """
        # Load data if path provided
        if isinstance(word_frequency_data, (str, Path)):
            self.df = pd.read_csv(word_frequency_data, sep=r"\s+", thousands=",")
        else:
            self.df = word_frequency_data.copy()

        self.word_column = word_column
        self.count_column = count_column

        # Add boundary markers to words
        self.df["word"] = (
            self.BOUNDARY + self.df[word_column].str.lower().astype(str) + self.BOUNDARY
        )

        # Build transition frequency dictionaries
        self._build_transition_counters()

        # Pre-calculate entropy and surprisal lookups
        self._construct_entropy_lookups()
        self._construct_surprisal_lookups()

    def _build_transition_counters(self):
        """Build character transition frequency dictionaries."""
        self.transitions: Dict[str, Counter[str]] = defaultdict(Counter)
        self.bigraph_to_char: Dict[str, Counter[str]] = defaultdict(Counter)
        self.trigraph_to_char: Dict[str, Counter[str]] = defaultdict(Counter)

        for row in self.df.itertuples():
            word = cast(str, row.word)
            count = getattr(row, self.count_column)

            for i in range(len(word) - 1):
                char1 = word[i]
                char2 = word[i + 1]
                self.transitions[char1][char2] += count

                if i > 0:
                    bigraph = word[i - 1 : i + 1]
                    self.bigraph_to_char[bigraph][char2] += count

                if i > 1:
                    trigraph = word[i - 2 : i + 1]
                    self.trigraph_to_char[trigraph][char2] += count

    def _construct_entropy_lookups(self):
        """Pre-calculate entropy values for all contexts."""

        def construct_entropy_lookup(
            frequency_dict: Dict[str, Counter[str]],
        ) -> Dict[str, float]:
            return {k: float(entropy(list(v.values()))) for k, v in frequency_dict.items()}

        self.char_char_entropy_lookup = construct_entropy_lookup(self.transitions)
        self.bigraph_char_entropy_lookup = construct_entropy_lookup(self.bigraph_to_char)
        self.trigraph_char_entropy_lookup = construct_entropy_lookup(self.trigraph_to_char)

    def _construct_surprisal_lookups(self):
        """Pre-calculate surprisal values for all context-target pairs."""

        def construct_surprisal_lookup(frequency_dict):
            surprisal_lookup = {}
            for context, targets in frequency_dict.items():
                total = sum(targets.values())
                surprisal_lookup[context] = {}
                for target, count in targets.items():
                    prob = count / total
                    surprisal_lookup[context][target] = -np.log2(prob)
            return surprisal_lookup

        self.char_char_surprisal_lookup = construct_surprisal_lookup(self.transitions)
        self.bigraph_char_surprisal_lookup = construct_surprisal_lookup(self.bigraph_to_char)
        self.trigraph_char_surprisal_lookup = construct_surprisal_lookup(self.trigraph_to_char)

    def calculate_metrics(self, text: str) -> Dict[str, float]:
        """Calculate character-level entropy and surprisal metrics for a text.

        Note:
            For consistent preprocessing, use `preprocess_text()` from
            `entroprisal.utils` before calling this method.

        Args:
            text: Input text string (should be lowercase, letters and spaces only)

        Returns:
            Dictionary with metrics:
                - char_entropy: Mean entropy for character transitions
                - char_surprisal: Mean surprisal for character transitions
                - bigraph_entropy: Mean entropy for bigraph contexts
                - bigraph_surprisal: Mean surprisal for bigraph contexts
                - trigraph_entropy: Mean entropy for trigraph contexts
                - trigraph_surprisal: Mean surprisal for trigraph contexts
                - char_entropy_support: Number of transitions with coverage
                - char_surprisal_support: Number of transitions with coverage
                - bigraph_entropy_support: Number of bigraphs with coverage
                - bigraph_surprisal_support: Number of bigraphs with coverage
                - trigraph_entropy_support: Number of trigraphs with coverage
                - trigraph_surprisal_support: Number of trigraphs with coverage
        """
        words = text.lower().split()

        char_char_entropies = []
        char_char_surprisals = []
        bigraph_char_entropies = []
        bigraph_char_surprisals = []
        trigraph_char_entropies = []
        trigraph_char_surprisals = []

        for word in words:
            word = self.BOUNDARY + word + self.BOUNDARY

            for i in range(len(word) - 1):
                char = word[i]
                next_char = word[i + 1]

                # Character-character entropy
                if char in self.char_char_entropy_lookup:
                    char_char_entropies.append(self.char_char_entropy_lookup[char])

                # Character-character surprisal
                if char in self.char_char_surprisal_lookup:
                    if next_char in self.char_char_surprisal_lookup[char]:
                        char_char_surprisals.append(
                            self.char_char_surprisal_lookup[char][next_char]
                        )

                # Bigraph entropy and surprisal
                if i > 0:
                    bigraph = word[i - 1 : i + 1]
                    if bigraph in self.bigraph_char_entropy_lookup:
                        bigraph_char_entropies.append(self.bigraph_char_entropy_lookup[bigraph])

                    if bigraph in self.bigraph_char_surprisal_lookup:
                        if next_char in self.bigraph_char_surprisal_lookup[bigraph]:
                            bigraph_char_surprisals.append(
                                self.bigraph_char_surprisal_lookup[bigraph][next_char]
                            )

                # Trigraph entropy and surprisal
                if i > 1:
                    trigraph = word[i - 2 : i + 1]
                    if trigraph in self.trigraph_char_entropy_lookup:
                        trigraph_char_entropies.append(self.trigraph_char_entropy_lookup[trigraph])

                    if trigraph in self.trigraph_char_surprisal_lookup:
                        if next_char in self.trigraph_char_surprisal_lookup[trigraph]:
                            trigraph_char_surprisals.append(
                                self.trigraph_char_surprisal_lookup[trigraph][next_char]
                            )

        metrics = {}

        # Character metrics
        if char_char_entropies:
            metrics["char_entropy"] = mean(char_char_entropies)
            metrics["char_entropy_support"] = len(char_char_entropies)

        if char_char_surprisals:
            metrics["char_surprisal"] = mean(char_char_surprisals)
            metrics["char_surprisal_support"] = len(char_char_surprisals)

        # Bigraph metrics
        if bigraph_char_entropies:
            metrics["bigraph_entropy"] = mean(bigraph_char_entropies)
            metrics["bigraph_entropy_support"] = len(bigraph_char_entropies)

        if bigraph_char_surprisals:
            metrics["bigraph_surprisal"] = mean(bigraph_char_surprisals)
            metrics["bigraph_surprisal_support"] = len(bigraph_char_surprisals)

        # Trigraph metrics
        if trigraph_char_entropies:
            metrics["trigraph_entropy"] = mean(trigraph_char_entropies)
            metrics["trigraph_entropy_support"] = len(trigraph_char_entropies)

        if trigraph_char_surprisals:
            metrics["trigraph_surprisal"] = mean(trigraph_char_surprisals)
            metrics["trigraph_surprisal_support"] = len(trigraph_char_surprisals)

        return metrics

    def calculate_batch(self, texts: List[str]) -> pd.DataFrame:
        """Calculate character-level entropy and surprisal metrics for multiple texts.

        Note:
            For consistent preprocessing, use `preprocess_text()` from
            `entroprisal.utils` before calling this method.

        Args:
            texts: List of text strings (should be lowercase, letters and spaces only)

        Returns:
            DataFrame with one row per text and columns for each metric
        """
        results = [self.calculate_metrics(text) for text in texts]
        return pd.DataFrame(results)

    def get_character_entropy(self, char: str) -> Optional[float]:
        """Get the entropy for a specific character.

        Args:
            char: Single character to look up

        Returns:
            Entropy value, or None if character not in reference corpus
        """
        return self.char_char_entropy_lookup.get(char.lower())

    def get_bigraph_entropy(self, bigraph: str) -> Optional[float]:
        """Get the entropy for a specific bigraph.

        Args:
            bigraph: Two-character string to look up

        Returns:
            Entropy value, or None if bigraph not in reference corpus
        """
        return self.bigraph_char_entropy_lookup.get(bigraph.lower())

    def get_trigraph_entropy(self, trigraph: str) -> Optional[float]:
        """Get the entropy for a specific trigraph.

        Args:
            trigraph: Three-character string to look up

        Returns:
            Entropy value, or None if trigraph not in reference corpus
        """
        return self.trigraph_char_entropy_lookup.get(trigraph.lower())

    def get_character_surprisal(self, context: str, target: str) -> Optional[float]:
        """Get the surprisal for a specific character transition.

        Args:
            context: Context character
            target: Target character

        Returns:
            Surprisal value, or None if transition not in reference corpus
        """
        context = context.lower()
        target = target.lower()
        if context in self.char_char_surprisal_lookup:
            return self.char_char_surprisal_lookup[context].get(target)
        return None

    def get_bigraph_surprisal(self, context: str, target: str) -> Optional[float]:
        """Get the surprisal for a specific bigraph transition.

        Args:
            context: Two-character context string
            target: Target character

        Returns:
            Surprisal value, or None if transition not in reference corpus
        """
        context = context.lower()
        target = target.lower()
        if context in self.bigraph_char_surprisal_lookup:
            return self.bigraph_char_surprisal_lookup[context].get(target)
        return None

    def get_trigraph_surprisal(self, context: str, target: str) -> Optional[float]:
        """Get the surprisal for a specific trigraph transition.

        Args:
            context: Three-character context string
            target: Target character

        Returns:
            Surprisal value, or None if transition not in reference corpus
        """
        context = context.lower()
        target = target.lower()
        if context in self.trigraph_char_surprisal_lookup:
            return self.trigraph_char_surprisal_lookup[context].get(target)
        return None
