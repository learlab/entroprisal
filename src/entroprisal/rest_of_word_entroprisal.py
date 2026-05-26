"""Rest-of-word entropy and surprisal calculator (bidirectional character-level).

This module calculates entropy and surprisal metrics for word completion in both
left-to-right and right-to-left directions, based on predicting the "rest of word"
given initial/final characters. This is a character-level analysis.
"""

import collections
import logging
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Union

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

    Per-position metrics are also exposed via `surprisal`, `entropy_reduction`, and
    `compute_all`, producing one row per word. Entropy reduction here is the clean
    Hale-style conditional mutual information form: because rest-of-word distributions
    are all over the same target (word identity), H(W | k-1 chars) - H(W | k chars)
    is the character-level counterpart of Hale (2016)'s parse-given-words formulation.
    """

    BOUNDARY = "#"

    # Reduction context lengths: n is the number of *content* characters in the full
    # prefix/suffix. n=1 uses the marginal word entropy H(W) as the Distribution A
    # baseline; n in {2, 3} use the (n-1)-char prefix entropy lookups.
    ENTROPY_REDUCTION_NS = (1, 2, 3)
    DIRECTIONS = ("lr", "rl")

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

        # Marginal word entropy H(W) = -sum_w p(w) log_2 p(w); used as the Distribution A
        # baseline for the n=1 entropy reduction.
        counts = list(self.word_frequencies.values())
        self.word_marginal_entropy = float(entropy(counts)) if counts else 0.0

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

    def calculate_metrics(self, tokens: List[str]) -> Dict[str, float]:
        """Calculate bidirectional rest-of-word entropy and surprisal metrics for a token list.

        Args:
            tokens: List of token strings (as returned by `preprocess_text()`).
                Tokens should be lowercase alphabetic strings.

        Returns:
            Dictionary with metrics:
                - mean_word_length: Average word length
                - lr_c{1,2,3}_entropy / lr_c{1,2,3}_surprisal: Left-to-right entropy
                  and surprisal at each conditioning prefix length
                - rl_c{1,2,3}_entropy / rl_c{1,2,3}_surprisal: Right-to-left analogues
                - lr_entropy_reduction_{1,2,3} / rl_entropy_reduction_{1,2,3}: Mean
                  Hale-style entropy reduction (clipped) by conditioning length;
                  see `entropy_reduction`. n=1 uses the marginal word entropy as
                  Distribution A baseline.
                - Support counts for each metric

            Per-position values are available via `compute_all`, `surprisal`, and
            `entropy_reduction`.

        Example:
            >>> from entroprisal import preprocess_text, RestOfWordEntropisalCalculator
            >>> tokens = preprocess_text("The quick brown fox")[0]
            >>> calc = RestOfWordEntropisalCalculator(words_df)
            >>> metrics = calc.calculate_metrics(tokens)
        """
        words = [token.lower() for token in tokens]

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

        # Per-position entropy_reduction, averaged over attested positions (clipped,
        # base 2). Mirrors the token-level pattern.
        if tokens:
            per_position = self._assemble(tokens, signed=False, base=2.0)
            for direction in self.DIRECTIONS:
                for n in self.ENTROPY_REDUCTION_NS:
                    col = f"{direction}_entropy_reduction_{n}"
                    values = per_position[col].dropna()
                    metrics[f"{col}_support"] = int(len(values))
                    if len(values) > 0:
                        metrics[col] = float(values.mean())

        return metrics

    def calculate_batch(self, token_lists: List[List[str]]) -> pd.DataFrame:
        """Calculate bidirectional rest-of-word metrics for multiple token lists.

        Args:
            token_lists: List of token lists (as returned by `preprocess_text()`).
                Each inner list contains lowercase alphabetic token strings.

        Returns:
            DataFrame with one row per token list and columns for each metric

        Example:
            >>> from entroprisal import preprocess_text, RestOfWordEntropisalCalculator
            >>> token_lists = preprocess_text(["First text.", "Second text."])
            >>> calc = RestOfWordEntropisalCalculator(words_df)
            >>> results_df = calc.calculate_batch(token_lists)
        """
        results = [self.calculate_metrics(tokens) for tokens in token_lists]
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

    # ------------------------------------------------------------------ per-position

    @staticmethod
    def _base_factor(base: float) -> float:
        """Conversion factor from base-2 information units to the requested base."""
        if base == 2.0:
            return 1.0
        return float(np.log(2.0) / np.log(base))

    def _lookup_rest_metrics(
        self,
        direction: str,
        n: int,
        padded: str,
    ) -> Dict[str, Optional[float]]:
        """Look up the (entropy, surprisal) pair for one direction and prefix length.

        For lr: prefix = padded[:n+1], rest = padded[n+1:].
        For rl: suffix = padded[-(n+1):], rest = padded[:-(n+1)].
        """
        if direction == "lr":
            prefix_len = n + 1
            context = padded[:prefix_len]
            rest = padded[prefix_len:]
        else:  # rl
            suffix_len = n + 1
            # padded[-suffix_len:] for suffix_len <= len(padded). For shorter strings,
            # this returns the full string; the resulting lookup miss handles it.
            context = padded[-suffix_len:]
            rest = padded[:-suffix_len]

        ent_lookup = getattr(self, f"{direction}_c{n}_entropy_lookup")
        surp_lookup = getattr(self, f"{direction}_c{n}_surprisal_lookup")

        ent = ent_lookup.get(context)
        surp = None
        inner = surp_lookup.get(context)
        if inner is not None:
            surp = inner.get(rest)

        return {"entropy": ent, "surprisal": surp}

    def _compute_per_position(self, tokens: List[str]) -> pd.DataFrame:
        """Build the raw per-word table (base-2, unclipped) for both directions.

        Emits one row per input token. For each direction (lr, rl) and each prefix
        length n in {1, 2, 3}, fills entropy and surprisal columns from the precomputed
        lookups. Unattested contexts (or words too short for a given prefix length)
        produce NaN.

        Returns one row per token with columns: token_index, word, then
        {lr,rl}_{entropy,surprisal}_{1,2,3} (12 metric columns).
        """
        rows = []
        for token_index, token in enumerate(tokens):
            padded = self.BOUNDARY + token.lower() + self.BOUNDARY
            row: Dict[str, object] = {"token_index": token_index, "word": token}
            for direction in self.DIRECTIONS:
                for n in (1, 2, 3):
                    metrics = self._lookup_rest_metrics(direction, n, padded)
                    row[f"{direction}_entropy_{n}"] = metrics["entropy"]
                    row[f"{direction}_surprisal_{n}"] = metrics["surprisal"]
            rows.append(row)

        columns = ["token_index", "word"]
        for direction in self.DIRECTIONS:
            for n in (1, 2, 3):
                columns.append(f"{direction}_entropy_{n}")
                columns.append(f"{direction}_surprisal_{n}")

        if not rows:
            return pd.DataFrame({col: pd.Series(dtype="float64") for col in columns})

        return pd.DataFrame(rows, columns=columns)

    def _assemble(self, tokens: List[str], signed: bool, base: float) -> pd.DataFrame:
        """Compute scaled, optionally-clipped per-word metrics for both directions.

        For each direction d in {lr, rl} and n in {1, 2, 3}, derives
        d_entropy_reduction_n with this Distribution A baseline:
          - n=1: word_marginal_entropy (H(W) over the full corpus)
          - n=2: d_entropy_1
          - n=3: d_entropy_2
        and the matching Distribution B is d_entropy_n. By the chain rule
        H(W | prefix_k) = H(rest_k | prefix_k), so these are clean
        H(X) - H(X|y) reductions over a fixed target (word identity).

        All defaults to max(., 0); pass signed=True to keep negative values.
        Availability flags are derived per metric (True only when every contributing
        context matched).
        """
        df = self._compute_per_position(tokens)
        factor = self._base_factor(base)

        # Scale entropy and surprisal columns, plus availability flags for surprisal.
        for direction in self.DIRECTIONS:
            for n in (1, 2, 3):
                df[f"{direction}_entropy_{n}"] = df[f"{direction}_entropy_{n}"] * factor
                df[f"{direction}_surprisal_{n}"] = df[f"{direction}_surprisal_{n}"] * factor
                df[f"{direction}_surprisal_{n}_available"] = df[
                    f"{direction}_surprisal_{n}"
                ].notna()

        marginal_scaled = self.word_marginal_entropy * factor

        clip_targets = []
        for direction in self.DIRECTIONS:
            for n in self.ENTROPY_REDUCTION_NS:
                col = f"{direction}_entropy_reduction_{n}"
                ent_n = df[f"{direction}_entropy_{n}"]
                if n == 1:
                    # Distribution A is the corpus-wide marginal H(W) (a scalar).
                    df[col] = marginal_scaled - ent_n
                    df[f"{col}_available"] = ent_n.notna()
                else:
                    ent_prev = df[f"{direction}_entropy_{n - 1}"]
                    df[col] = ent_prev - ent_n
                    df[f"{col}_available"] = ent_prev.notna() & ent_n.notna()
                clip_targets.append(col)

        if not signed:
            for col in clip_targets:
                # `< 0` is False for NaN, so NaN flows through unchanged.
                df.loc[df[col] < 0, col] = 0.0

        return df

    def surprisal(
        self,
        tokens: List[str],
        *,
        direction: str = "lr",
        n: int = 2,
        base: float = 2.0,
    ) -> pd.DataFrame:
        """Per-word rest-of-word surprisal: -log P(rest | n-char prefix/suffix).

        For lr direction, returns surprisal of the rest of the word given the first
        n content characters (and the leading boundary). For rl, given the last n
        content characters (and the trailing boundary). Positions whose prefix is
        unattested -- or the specific rest sequence is unseen -- get NaN and
        `surprisal_available == False` (no backoff).

        Args:
            tokens: List of token strings (e.g. from `preprocess_text(text)[0]`).
            direction: "lr" (default) or "rl".
            n: Conditioning prefix length, 1, 2, or 3 (default 2).
            base: Logarithm base (default 2.0 for bits).

        Returns:
            DataFrame with columns: token_index, word, surprisal, surprisal_available.
        """
        self._validate_direction(direction)
        if n not in (1, 2, 3):
            raise ValueError(f"n must be one of (1, 2, 3), got {n!r}")
        df = self._assemble(tokens, signed=True, base=base)
        return (
            df[
                [
                    "token_index",
                    "word",
                    f"{direction}_surprisal_{n}",
                    f"{direction}_surprisal_{n}_available",
                ]
            ]
            .rename(
                columns={
                    f"{direction}_surprisal_{n}": "surprisal",
                    f"{direction}_surprisal_{n}_available": "surprisal_available",
                }
            )
            .copy()
        )

    def entropy_reduction(
        self,
        tokens: List[str],
        *,
        direction: str = "lr",
        n: int = 2,
        signed: bool = False,
        base: float = 2.0,
    ) -> pd.DataFrame:
        """Per-word rest-of-word entropy reduction (Hale-style CMI over word identity).

        For conditioning prefix length ``n``, computes
        H(W | (n-1)-char context) - H(W | n-char context): how much observing one
        additional character reduced uncertainty about the word's identity. Because
        the conditioning random variable is the same (word identity) across prefix
        lengths -- by the chain rule, H(W | prefix_k) = H(rest_k | prefix_k) -- this
        is a clean Hale-style conditional mutual information, directly parallel to
        Hale (2016)'s parse-given-words formulation but with characters reducing
        uncertainty about word identity.

        - n=1 (default Distribution A is the marginal): H(W) - H(W | first char).
          This is the most theoretically interesting reduction: how much the first
          observed character narrows the space of possible words.
        - n=2: H(W | first char) - H(W | first 2 chars).
        - n=3: H(W | first 2 chars) - H(W | first 3 chars).

        For direction="rl", the same logic applies with suffix characters instead.
        No smoothing is applied.

        Args:
            tokens: List of token strings.
            direction: "lr" (default) or "rl".
            n: Conditioning prefix length, 1, 2, or 3 (default 2).
            signed: If False (default), return max(reduction, 0). If True, return the
                signed value.
            base: Logarithm base (default 2.0 for bits).

        Returns:
            DataFrame with columns: token_index, word, entropy_reduction, available.
            NaN where either contributing context is unattested.
        """
        self._validate_direction(direction)
        if n not in self.ENTROPY_REDUCTION_NS:
            raise ValueError(f"n must be one of {self.ENTROPY_REDUCTION_NS}, got {n!r}")
        df = self._assemble(tokens, signed=signed, base=base)
        return (
            df[
                [
                    "token_index",
                    "word",
                    f"{direction}_entropy_reduction_{n}",
                    f"{direction}_entropy_reduction_{n}_available",
                ]
            ]
            .rename(
                columns={
                    f"{direction}_entropy_reduction_{n}": "entropy_reduction",
                    f"{direction}_entropy_reduction_{n}_available": "available",
                }
            )
            .copy()
        )

    def compute_all(
        self, tokens: List[str], *, signed: bool = False, base: float = 2.0
    ) -> pd.DataFrame:
        """Compute all per-word rest-of-word metrics in one pass.

        Convenience method for comparative analysis. See `surprisal` and
        `entropy_reduction` for the definitions.

        Args:
            tokens: List of token strings.
            signed: Applies to entropy_reduction (default clipped).
            base: Logarithm base (default 2.0 for bits).

        Returns:
            DataFrame with one row per word: token_index, word, then for each
            direction d in {lr, rl}: d_surprisal_{1,2,3}, d_entropy_reduction_{1,2,3},
            and a matching ``*_available`` flag for each metric.
        """
        df = self._assemble(tokens, signed=signed, base=base)
        metric_cols = []
        flag_cols = []
        for direction in self.DIRECTIONS:
            for n in (1, 2, 3):
                metric_cols.append(f"{direction}_surprisal_{n}")
                flag_cols.append(f"{direction}_surprisal_{n}_available")
            for n in self.ENTROPY_REDUCTION_NS:
                metric_cols.append(f"{direction}_entropy_reduction_{n}")
                flag_cols.append(f"{direction}_entropy_reduction_{n}_available")
        return df[["token_index", "word", *metric_cols, *flag_cols]].copy()

    def _validate_direction(self, direction: str) -> None:
        if direction not in self.DIRECTIONS:
            raise ValueError(f"direction must be one of {self.DIRECTIONS}, got {direction!r}")
