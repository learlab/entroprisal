"""Rest-of-word entropy and surprisal calculator (bidirectional character-level).

This module calculates entropy and surprisal metrics for word completion in both
left-to-right and right-to-left directions, based on predicting the "rest of word"
given initial/final characters. This is a character-level analysis.
"""

import collections
import logging
from pathlib import Path
from statistics import mean
from typing import Dict, List, Union

import numpy as np
import pandas as pd

from .utils import entropy

logger = logging.getLogger(__name__)


class RestOfWordEntropisalCalculator:
    """Calculate bidirectional rest-of-word entropy and surprisal metrics (character-level).

    This calculator computes both entropy and surprisal for predicting the rest of a word given:
    - Left-to-right (lr): First n characters predict the rest
    - Right-to-left (rl): Last n characters predict the rest

    For example, given "#for#":
    - lr_c1: Given "#", what are the entropy/surprisal for "for#"?
    - rl_c1: Given "#", what are the entropy/surprisal for "#for"?
    - lr_c2: Given "#f", what are the entropy/surprisal for "or#"?
    - rl_c2: Given "r#", what are the entropy/surprisal for "#fo"?

    Words are processed with boundary markers (#) at start and end.
    This is a character-level analysis, not word-level.
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

        # Build frequency dictionaries
        self._build_frequency_counters()

        # Pre-calculate entropy and surprisal lookups
        self._construct_entropy_lookups()
        self._construct_surprisal_lookups()

    def _build_frequency_counters(self):
        """Build frequency counters for left-to-right and right-to-left predictions.

        Creates counters for:
        - lr_c1, lr_c2, lr_c3: Left-to-right conditioned on 1, 2, 3 chars
        - rl_c1, rl_c2, rl_c3: Right-to-left conditioned on 1, 2, 3 chars
        """
        self.word_frequencies = collections.Counter()
        self.lr_c1 = collections.defaultdict(collections.Counter)
        self.lr_c2 = collections.defaultdict(collections.Counter)
        self.lr_c3 = collections.defaultdict(collections.Counter)
        self.rl_c1 = collections.defaultdict(collections.Counter)
        self.rl_c2 = collections.defaultdict(collections.Counter)
        self.rl_c3 = collections.defaultdict(collections.Counter)

        for row in self.df.itertuples():
            word = row.word
            freq = getattr(row, self.count_column)

            assert isinstance(word, str), "Word must be a string"
            # Store word frequency
            self.word_frequencies[word] = freq

            # Left-to-right: first n chars predict rest of word
            # p("for#"|"#")
            self.lr_c1[word[:2]][word[2:]] += freq

            # Right-to-left: last n chars predict rest of word
            # p("#for"|"#")
            self.rl_c1[word[-2:]][word[:-2]] += freq

            # p("or#"|"#f")
            self.lr_c2[word[:3]][word[3:]] += freq
            # p("#fo"|"r#")
            self.rl_c2[word[-3:]][word[:-3]] += freq

            # Cannot be calculated for single-letter words like "#a#"
            if len(word) >= 4:
                # p("r#"|"#fo")
                self.lr_c3[word[:4]][word[4:]] += freq
                # p("#f"|"or#")
                self.rl_c3[word[-4:]][word[:-4]] += freq

    def _construct_entropy_lookups(self):
        """Pre-calculate entropy values for all contexts."""

        def construct_entropy_lookup(frequency_dict):
            return {k: entropy(list(v.values())) for k, v in frequency_dict.items()}

        self.lr_c1_entropy_lookup = construct_entropy_lookup(self.lr_c1)
        self.lr_c2_entropy_lookup = construct_entropy_lookup(self.lr_c2)
        self.lr_c3_entropy_lookup = construct_entropy_lookup(self.lr_c3)
        self.rl_c1_entropy_lookup = construct_entropy_lookup(self.rl_c1)
        self.rl_c2_entropy_lookup = construct_entropy_lookup(self.rl_c2)
        self.rl_c3_entropy_lookup = construct_entropy_lookup(self.rl_c3)

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

        self.lr_c1_surprisal_lookup = construct_surprisal_lookup(self.lr_c1)
        self.lr_c2_surprisal_lookup = construct_surprisal_lookup(self.lr_c2)
        self.lr_c3_surprisal_lookup = construct_surprisal_lookup(self.lr_c3)
        self.rl_c1_surprisal_lookup = construct_surprisal_lookup(self.rl_c1)
        self.rl_c2_surprisal_lookup = construct_surprisal_lookup(self.rl_c2)
        self.rl_c3_surprisal_lookup = construct_surprisal_lookup(self.rl_c3)

    def calculate_metrics(self, text: str, preprocess: bool = True) -> Dict[str, float]:
        """Calculate bidirectional rest-of-word entropy and surprisal metrics for text.

        Args:
            text: Input text string
            preprocess: Whether to apply basic preprocessing (lowercase, remove non-letters)

        Returns:
            Dictionary with metrics:
                - mean_word_length: Average word length
                - lr_c1_entropy: Left-to-right entropy (1 char context)
                - lr_c1_surprisal: Left-to-right surprisal (1 char context)
                - lr_c2_entropy: Left-to-right entropy (2 char context)
                - lr_c2_surprisal: Left-to-right surprisal (2 char context)
                - lr_c3_entropy: Left-to-right entropy (3 char context)
                - lr_c3_surprisal: Left-to-right surprisal (3 char context)
                - rl_c1_entropy: Right-to-left entropy (1 char context)
                - rl_c1_surprisal: Right-to-left surprisal (1 char context)
                - rl_c2_entropy: Right-to-left entropy (2 char context)
                - rl_c2_surprisal: Right-to-left surprisal (2 char context)
                - rl_c3_entropy: Right-to-left entropy (3 char context)
                - rl_c3_surprisal: Right-to-left surprisal (3 char context)
                - Support counts for each metric
        """
        # Preprocess if requested
        if preprocess:
            import re

            not_letter_not_space = re.compile("[^a-z ]+")
            text = not_letter_not_space.sub("", text.lower())

        words = text.split()

        lr_c1_entropies = []
        lr_c1_surprisals = []
        lr_c2_entropies = []
        lr_c2_surprisals = []
        lr_c3_entropies = []
        lr_c3_surprisals = []
        rl_c1_entropies = []
        rl_c1_surprisals = []
        rl_c2_entropies = []
        rl_c2_surprisals = []
        rl_c3_entropies = []
        rl_c3_surprisals = []
        word_lengths = []

        for word in words:
            word_lengths.append(len(word))
            word = self.BOUNDARY + word + self.BOUNDARY

            # Left-to-right c1 (1 char context)
            if word[:2] in self.lr_c1_entropy_lookup:
                lr_c1_entropies.append(self.lr_c1_entropy_lookup[word[:2]])

            if word[:2] in self.lr_c1_surprisal_lookup:
                rest = word[2:]
                if rest in self.lr_c1_surprisal_lookup[word[:2]]:
                    lr_c1_surprisals.append(self.lr_c1_surprisal_lookup[word[:2]][rest])

            # Left-to-right c2 (2 char context)
            if word[:3] in self.lr_c2_entropy_lookup:
                lr_c2_entropies.append(self.lr_c2_entropy_lookup[word[:3]])

            if word[:3] in self.lr_c2_surprisal_lookup:
                rest = word[3:]
                if rest in self.lr_c2_surprisal_lookup[word[:3]]:
                    lr_c2_surprisals.append(self.lr_c2_surprisal_lookup[word[:3]][rest])

            # Left-to-right c3 (3 char context)
            if len(word) >= 4:
                if word[:4] in self.lr_c3_entropy_lookup:
                    lr_c3_entropies.append(self.lr_c3_entropy_lookup[word[:4]])

                if word[:4] in self.lr_c3_surprisal_lookup:
                    rest = word[4:]
                    if rest in self.lr_c3_surprisal_lookup[word[:4]]:
                        lr_c3_surprisals.append(self.lr_c3_surprisal_lookup[word[:4]][rest])

            # Right-to-left c1 (1 char context)
            if word[-2:] in self.rl_c1_entropy_lookup:
                rl_c1_entropies.append(self.rl_c1_entropy_lookup[word[-2:]])

            if word[-2:] in self.rl_c1_surprisal_lookup:
                rest = word[:-2]
                if rest in self.rl_c1_surprisal_lookup[word[-2:]]:
                    rl_c1_surprisals.append(self.rl_c1_surprisal_lookup[word[-2:]][rest])

            # Right-to-left c2 (2 char context)
            if word[-3:] in self.rl_c2_entropy_lookup:
                rl_c2_entropies.append(self.rl_c2_entropy_lookup[word[-3:]])

            if word[-3:] in self.rl_c2_surprisal_lookup:
                rest = word[:-3]
                if rest in self.rl_c2_surprisal_lookup[word[-3:]]:
                    rl_c2_surprisals.append(self.rl_c2_surprisal_lookup[word[-3:]][rest])

            # Right-to-left c3 (3 char context)
            if len(word) >= 4:
                if word[-4:] in self.rl_c3_entropy_lookup:
                    rl_c3_entropies.append(self.rl_c3_entropy_lookup[word[-4:]])

                if word[-4:] in self.rl_c3_surprisal_lookup:
                    rest = word[:-4]
                    if rest in self.rl_c3_surprisal_lookup[word[-4:]]:
                        rl_c3_surprisals.append(self.rl_c3_surprisal_lookup[word[-4:]][rest])

        metrics = {}

        # Word length
        if word_lengths:
            metrics["mean_word_length"] = mean(word_lengths)

        # Left-to-right metrics
        if lr_c1_entropies:
            metrics["lr_c1_entropy"] = mean(lr_c1_entropies)
            metrics["lr_c1_entropy_support"] = len(lr_c1_entropies)

        if lr_c1_surprisals:
            metrics["lr_c1_surprisal"] = mean(lr_c1_surprisals)
            metrics["lr_c1_surprisal_support"] = len(lr_c1_surprisals)

        if lr_c2_entropies:
            metrics["lr_c2_entropy"] = mean(lr_c2_entropies)
            metrics["lr_c2_entropy_support"] = len(lr_c2_entropies)

        if lr_c2_surprisals:
            metrics["lr_c2_surprisal"] = mean(lr_c2_surprisals)
            metrics["lr_c2_surprisal_support"] = len(lr_c2_surprisals)

        if lr_c3_entropies:
            metrics["lr_c3_entropy"] = mean(lr_c3_entropies)
            metrics["lr_c3_entropy_support"] = len(lr_c3_entropies)

        if lr_c3_surprisals:
            metrics["lr_c3_surprisal"] = mean(lr_c3_surprisals)
            metrics["lr_c3_surprisal_support"] = len(lr_c3_surprisals)

        # Right-to-left metrics
        if rl_c1_entropies:
            metrics["rl_c1_entropy"] = mean(rl_c1_entropies)
            metrics["rl_c1_entropy_support"] = len(rl_c1_entropies)

        if rl_c1_surprisals:
            metrics["rl_c1_surprisal"] = mean(rl_c1_surprisals)
            metrics["rl_c1_surprisal_support"] = len(rl_c1_surprisals)

        if rl_c2_entropies:
            metrics["rl_c2_entropy"] = mean(rl_c2_entropies)
            metrics["rl_c2_entropy_support"] = len(rl_c2_entropies)

        if rl_c2_surprisals:
            metrics["rl_c2_surprisal"] = mean(rl_c2_surprisals)
            metrics["rl_c2_surprisal_support"] = len(rl_c2_surprisals)

        if rl_c3_entropies:
            metrics["rl_c3_entropy"] = mean(rl_c3_entropies)
            metrics["rl_c3_entropy_support"] = len(rl_c3_entropies)

        if rl_c3_surprisals:
            metrics["rl_c3_surprisal"] = mean(rl_c3_surprisals)
            metrics["rl_c3_surprisal_support"] = len(rl_c3_surprisals)

        return metrics

    def calculate_batch(self, texts: List[str], preprocess: bool = True) -> pd.DataFrame:
        """Calculate bidirectional rest-of-word metrics for multiple texts.

        Args:
            texts: List of text strings
            preprocess: Whether to apply basic preprocessing

        Returns:
            DataFrame with one row per text and columns for each metric
        """
        results = [self.calculate_metrics(text, preprocess=preprocess) for text in texts]
        return pd.DataFrame(results)

    def get_word_frequency(self, word: str) -> int:
        """Get the frequency of a specific word in the reference corpus.

        Args:
            word: Word to look up (will be lowercased and boundaries added)

        Returns:
            Frequency count, or 0 if word not in corpus
        """
        word_with_boundaries = self.BOUNDARY + word.lower() + self.BOUNDARY
        return self.word_frequencies.get(word_with_boundaries, 0)
