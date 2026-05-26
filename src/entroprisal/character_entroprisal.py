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

    # Conditioning context lengths matching the token-level convention: n is the number
    # of preceding characters in the full distribution. Entropy reduction needs to drop
    # one preceding char, so it is defined for n in {2, 3}; surprisal and entropy
    # difference are plain next-char comparisons, defined for n in {1, 2, 3}.
    SURPRISAL_NS = (1, 2, 3)
    ENTROPY_REDUCTION_NS = (2, 3)
    ENTROPY_DIFFERENCE_NS = (1, 2, 3)

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
        self._build_gap_transition_counters()

        # Pre-calculate entropy and surprisal lookups
        self._construct_entropy_lookups()
        self._construct_surprisal_lookups()
        self._construct_gap_entropy_lookups()

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

    def _build_gap_transition_counters(self):
        """Build gap-context transition counters for entropy reduction (Distribution A).

        For each entropy-reduction context length n, Distribution A conditions on the
        n-1 chars ending at c_{i-2} -- the full n-char context with the most recent
        char c_{i-1} dropped -- and marginalizes over c_{i-1} by summing counts.
        Summing counts across the dropped position *is* the marginalization.

        - gap_char_to_char (n=2): c_{i-2} -> c_i (1-char context with a 1-char gap).
        - gap_bigraph_to_char (n=3): (c_{i-3}, c_{i-2}) -> c_i (2-char context with
          a 1-char gap). Distinct from the contiguous bigraph_to_char.
        """
        self.gap_char_to_char: Dict[str, Counter[str]] = defaultdict(Counter)
        self.gap_bigraph_to_char: Dict[str, Counter[str]] = defaultdict(Counter)

        for row in self.df.itertuples():
            word = cast(str, row.word)
            count = getattr(row, self.count_column)

            for i in range(2, len(word)):
                # n=2: gap context is the single char at i-2 (skip i-1).
                self.gap_char_to_char[word[i - 2]][word[i]] += count

                if i >= 3:
                    # n=3: gap context is the bigraph (i-3, i-2) (skip i-1).
                    self.gap_bigraph_to_char[word[i - 3 : i - 1]][word[i]] += count

    def _construct_entropy_lookups(self):
        """Pre-calculate entropy values for all contexts."""

        def construct_entropy_lookup(
            frequency_dict: Dict[str, Counter[str]],
        ) -> Dict[str, float]:
            return {k: float(entropy(list(v.values()))) for k, v in frequency_dict.items()}

        self.char_char_entropy_lookup = construct_entropy_lookup(self.transitions)
        self.bigraph_char_entropy_lookup = construct_entropy_lookup(self.bigraph_to_char)
        self.trigraph_char_entropy_lookup = construct_entropy_lookup(self.trigraph_to_char)

    def _construct_gap_entropy_lookups(self):
        """Pre-calculate gap-context entropy values (Distribution A) for entropy reduction.

        No smoothing is applied: raw MLE entropies over the (unfiltered) counts. Stored
        in base 2; callers rescale for other bases. Matches the token-level pattern in
        `TokenEntropisalCalculator._build_gap_entropy_lookups`.
        """

        def construct(frequency_dict: Dict[str, Counter[str]]) -> Dict[str, float]:
            return {k: float(entropy(list(v.values()))) for k, v in frequency_dict.items()}

        self.gap_char_entropy_lookup = construct(self.gap_char_to_char)
        self.gap_bigraph_entropy_lookup = construct(self.gap_bigraph_to_char)

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

    def calculate_metrics(self, tokens: List[str]) -> Dict[str, float]:
        """Calculate character-level entropy and surprisal metrics for a token list.

        Args:
            tokens: List of token strings (as returned by `preprocess_text()`).
                Tokens should be lowercase alphabetic strings.

        Returns:
            Dictionary with metrics:
                - char_entropy: Mean entropy for character transitions
                - char_surprisal: Mean surprisal for character transitions
                - bigraph_entropy: Mean entropy for bigraph contexts
                - bigraph_surprisal: Mean surprisal for bigraph contexts
                - trigraph_entropy: Mean entropy for trigraph contexts
                - trigraph_surprisal: Mean surprisal for trigraph contexts
                - char_entropy_reduction_{2,3}: Mean conditional-mutual-information
                  reduction (clipped) over attested positions, by conditioning
                  context length n; see `entropy_reduction`
                - char_entropy_difference_{1,2,3}: Mean Lowder-style entropy
                  difference (clipped) over attested positions; see `entropy_difference`
                - *_support: Number of positions contributing to each mean

            Per-position values are available via `compute_all`, `surprisal`,
            `entropy_reduction`, and `entropy_difference`.

        Example:
            >>> from entroprisal import preprocess_text, CharacterEntropisalCalculator
            >>> tokens = preprocess_text("The quick brown fox")[0]
            >>> calc = CharacterEntropisalCalculator(words_df)
            >>> metrics = calc.calculate_metrics(tokens)
        """
        words = [token.lower() for token in tokens]

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

        # Per-position entropy_reduction / entropy_difference, averaged over attested
        # positions (clipped, base 2). Mirrors TokenEntropisalCalculator.calculate_metrics.
        if tokens:
            per_position = self._assemble(tokens, signed=False, base=2.0)
            for n in self.ENTROPY_REDUCTION_NS:
                col = f"entropy_reduction_{n}"
                values = per_position[col].dropna()
                metrics[f"char_{col}_support"] = int(len(values))
                if len(values) > 0:
                    metrics[f"char_{col}"] = float(values.mean())
            for n in self.ENTROPY_DIFFERENCE_NS:
                col = f"entropy_difference_{n}"
                values = per_position[col].dropna()
                metrics[f"char_{col}_support"] = int(len(values))
                if len(values) > 0:
                    metrics[f"char_{col}"] = float(values.mean())

        return metrics

    def calculate_batch(self, token_lists: List[List[str]]) -> pd.DataFrame:
        """Calculate character-level entropy and surprisal metrics for multiple token lists.

        Args:
            token_lists: List of token lists (as returned by `preprocess_text()`).
                Each inner list contains lowercase alphabetic token strings.

        Returns:
            DataFrame with one row per token list and columns for each metric

        Example:
            >>> from entroprisal import preprocess_text, CharacterEntropisalCalculator
            >>> token_lists = preprocess_text(["First text.", "Second text."])
            >>> calc = CharacterEntropisalCalculator(words_df)
            >>> results_df = calc.calculate_batch(token_lists)
        """
        results = [self.calculate_metrics(tokens) for tokens in token_lists]
        return pd.DataFrame(results)

    # ------------------------------------------------------------------ per-position

    @staticmethod
    def _base_factor(base: float) -> float:
        """Conversion factor from base-2 information units to the requested base."""
        if base == 2.0:
            return 1.0
        return float(np.log(2.0) / np.log(base))

    def _lookup_entropy(self, n: int, key: Optional[str]) -> Optional[float]:
        """Look up the n-char full-context entropy for `key`, returning None if missing."""
        if key is None:
            return None
        if n == 1:
            return self.char_char_entropy_lookup.get(key)
        if n == 2:
            return self.bigraph_char_entropy_lookup.get(key)
        if n == 3:
            return self.trigraph_char_entropy_lookup.get(key)
        raise ValueError(f"unsupported entropy context length n={n}")

    def _lookup_gap_entropy(self, n: int, key: Optional[str]) -> Optional[float]:
        """Look up the gap (Distribution A) entropy at context length n."""
        if key is None:
            return None
        if n == 2:
            return self.gap_char_entropy_lookup.get(key)
        if n == 3:
            return self.gap_bigraph_entropy_lookup.get(key)
        raise ValueError(f"unsupported gap context length n={n}")

    def _lookup_surprisal(self, n: int, context: Optional[str], target: str) -> Optional[float]:
        """Look up surprisal for `target` given `context` at context length n."""
        if context is None:
            return None
        if n == 1:
            tbl = self.char_char_surprisal_lookup
        elif n == 2:
            tbl = self.bigraph_char_surprisal_lookup
        elif n == 3:
            tbl = self.trigraph_char_surprisal_lookup
        else:
            raise ValueError(f"unsupported surprisal context length n={n}")
        inner = tbl.get(context)
        if inner is None:
            return None
        return inner.get(target)

    def _compute_per_position(self, tokens: List[str]) -> pd.DataFrame:
        """Build the raw per-position table (base-2, unclipped) for char transitions.

        Emits one row per target character position within each boundary-padded word.
        For target c_i, ent_{n} = H(c_i | c_{i-n}..c_{i-1}) is the full-context entropy
        at length n (both Distribution B for entropy reduction and the term differenced
        across positions for entropy difference). gap_{n} is the matching Distribution A
        from the gap entropy lookups. surprisal_{n} is the surprisal of the target
        given n preceding characters; callers select the desired n. Unattested or
        out-of-range contexts produce NaN (no backoff).

        ent_prev_{n} is the previous *within-word* position's ent_{n}; it is reset to
        NaN at the start of each new word so entropy_difference never crosses word
        boundaries (matching the within-word loop semantics of `calculate_metrics`).

        Returns one row per (token_index, position) with columns: token_index, word,
        position, target, surprisal_{1,2,3}, ent_{1,2,3}, ent_prev_{1,2,3}, gap_{2,3}.
        """
        ent_ns = sorted(set(self.ENTROPY_DIFFERENCE_NS) | set(self.ENTROPY_REDUCTION_NS))

        rows = []
        for token_index, token in enumerate(tokens):
            word = self.BOUNDARY + token.lower() + self.BOUNDARY
            # Track previous within-word ent_{n} for the shift used in entropy_difference.
            prev_ent: Dict[int, Optional[float]] = {n: None for n in ent_ns}

            for i in range(1, len(word)):
                target = word[i]
                row: Dict[str, object] = {
                    "token_index": token_index,
                    "word": token,
                    "position": i,
                    "target": target,
                }

                # Full-context entropy at each length and the previous position's value.
                for n in ent_ns:
                    bkey = word[i - n : i] if i >= n else None
                    ent = self._lookup_entropy(n, bkey)
                    row[f"ent_{n}"] = ent
                    row[f"ent_prev_{n}"] = prev_ent[n]
                    prev_ent[n] = ent

                # Gap (Distribution A) entropy at each entropy-reduction context length.
                for n in self.ENTROPY_REDUCTION_NS:
                    akey = word[i - n : i - 1] if i >= n else None
                    row[f"gap_{n}"] = self._lookup_gap_entropy(n, akey)

                # Surprisal at each context length n: -log P(c_i | n preceding chars).
                for n in self.SURPRISAL_NS:
                    bkey = word[i - n : i] if i >= n else None
                    row[f"surprisal_{n}"] = self._lookup_surprisal(n, bkey, target)

                rows.append(row)

        # Build column schema explicitly so empty inputs still yield expected columns.
        columns = ["token_index", "word", "position", "target"]
        for n in self.SURPRISAL_NS:
            columns.append(f"surprisal_{n}")
        for n in ent_ns:
            columns.append(f"ent_{n}")
            columns.append(f"ent_prev_{n}")
        for n in self.ENTROPY_REDUCTION_NS:
            columns.append(f"gap_{n}")

        if not rows:
            return pd.DataFrame({col: pd.Series(dtype="float64") for col in columns})

        return pd.DataFrame(rows, columns=columns)

    def _assemble(self, tokens: List[str], signed: bool, base: float) -> pd.DataFrame:
        """Compute scaled, optionally-clipped per-position metrics for all context lengths.

        entropy_reduction_{n} = H(A_n) - H(B_n)  (conditional mutual information form).
        entropy_difference_{n} = E_n[i-1] - E_n[i]  (Lowder-style, distinct random vars,
        within-word only). All default to max(., 0); pass signed=True to keep negative
        values. Availability flags are derived per metric (True only when every
        contributing context matched in the reference corpus).
        """
        df = self._compute_per_position(tokens)
        factor = self._base_factor(base)

        # Scale surprisal at each context length and derive availability flags.
        for n in self.SURPRISAL_NS:
            df[f"surprisal_{n}"] = df[f"surprisal_{n}"] * factor
            df[f"surprisal_{n}_available"] = df[f"surprisal_{n}"].notna()

        clip_targets = []

        for n in self.ENTROPY_REDUCTION_NS:
            col = f"entropy_reduction_{n}"
            df[col] = (df[f"gap_{n}"] - df[f"ent_{n}"]) * factor
            df[f"{col}_available"] = df[f"gap_{n}"].notna() & df[f"ent_{n}"].notna()
            clip_targets.append(col)

        for n in self.ENTROPY_DIFFERENCE_NS:
            col = f"entropy_difference_{n}"
            df[col] = (df[f"ent_prev_{n}"] - df[f"ent_{n}"]) * factor
            df[f"{col}_available"] = df[f"ent_prev_{n}"].notna() & df[f"ent_{n}"].notna()
            clip_targets.append(col)

        if not signed:
            for col in clip_targets:
                # `< 0` is False for NaN, so NaN flows through unchanged.
                df.loc[df[col] < 0, col] = 0.0

        return df

    def surprisal(self, tokens: List[str], *, n: int = 3, base: float = 2.0) -> pd.DataFrame:
        """Per-position character surprisal: -log P(c_i | n preceding chars).

        The information value of each character given the n preceding characters within
        the boundary-padded word. Positions whose context is unattested -- or which lack
        n preceding chars -- get NaN and `surprisal_available == False` (no backoff).

        - n=3 (default, trigraph): -log P(c_i | c_{i-3}, c_{i-2}, c_{i-1}).
        - n=2 (bigraph):            -log P(c_i | c_{i-2}, c_{i-1}).
        - n=1 (single char):        -log P(c_i | c_{i-1}).

        Args:
            tokens: List of token strings (e.g. from `preprocess_text(text)[0]`).
            n: Conditioning context length, 1, 2, or 3 (default 3).
            base: Logarithm base (default 2.0 for bits).

        Returns:
            DataFrame with columns: token_index, word, position, target, surprisal,
            surprisal_available. One row per target character position (position 0 is
            the leading boundary and never a target).
        """
        if n not in self.SURPRISAL_NS:
            raise ValueError(f"n must be one of {self.SURPRISAL_NS}, got {n!r}")
        df = self._assemble(tokens, signed=True, base=base)
        out = df[
            [
                "token_index",
                "word",
                "position",
                "target",
                f"surprisal_{n}",
                f"surprisal_{n}_available",
            ]
        ].copy()
        out = out.rename(
            columns={
                f"surprisal_{n}": "surprisal",
                f"surprisal_{n}_available": "surprisal_available",
            }
        )
        return out

    def entropy_reduction(
        self, tokens: List[str], *, n: int = 3, signed: bool = False, base: float = 2.0
    ) -> pd.DataFrame:
        """Per-position character entropy reduction (conditional mutual information form).

        For conditioning context length ``n``, computes
        H(c_i | c_{i-n}..c_{i-2}) - H(c_i | c_{i-n}..c_{i-1}): how much observing the
        most recent preceding character c_{i-1} reduced uncertainty about the target c_i.

        - n=3 (default, trigraph): H(c_i | c_{i-3}, c_{i-2}) - H(c_i | c_{i-3..i-1}).
        - n=2 (bigraph):            H(c_i | c_{i-2})         - H(c_i | c_{i-2}, c_{i-1}).

        This is the character-level counterpart of `TokenEntropisalCalculator.entropy_reduction`.
        Distribution A pools across more continuations than Distribution B, so when the
        dropped character is deterministic given the earlier context, A == B and the
        reduction is exactly 0. No smoothing is applied.

        Args:
            tokens: List of token strings.
            n: Conditioning context length, 2 (bigraph) or 3 (trigraph, default).
            signed: If False (default), return max(reduction, 0). If True, return the
                signed value (negative means the new context broadened expectations).
            base: Logarithm base (default 2.0 for bits).

        Returns:
            DataFrame with columns: token_index, word, position, target,
            entropy_reduction, available. NaN where either contributing context is
            unattested.

        See also `entropy_difference` for the Lowder-style version.
        """
        if n not in self.ENTROPY_REDUCTION_NS:
            raise ValueError(f"n must be one of {self.ENTROPY_REDUCTION_NS}, got {n!r}")
        df = self._assemble(tokens, signed=signed, base=base)
        out = df[
            [
                "token_index",
                "word",
                "position",
                "target",
                f"entropy_reduction_{n}",
                f"entropy_reduction_{n}_available",
            ]
        ].copy()
        out = out.rename(
            columns={
                f"entropy_reduction_{n}": "entropy_reduction",
                f"entropy_reduction_{n}_available": "available",
            }
        )
        return out

    def entropy_difference(
        self, tokens: List[str], *, n: int = 3, signed: bool = False, base: float = 2.0
    ) -> pd.DataFrame:
        """Per-position character entropy difference (Lowder-style), clipped at 0 by default.

        For conditioning context length ``n``, computes E_n[i-1] - E_n[i], where
        E_n[i] = H(c_i | c_{i-n}..c_{i-1}) is the next-char entropy at position i.
        Differences are taken within a single word -- the first scorable position in
        each word has no predecessor and yields NaN. Unlike `entropy_reduction`, this
        differences entropies over *distinct random variables* (guesses at adjacent
        char positions), not H(X) - H(X|y) over a fixed target.

        Args:
            tokens: List of token strings.
            n: Conditioning context length, 1 (single-char), 2 (bigraph), or 3
                (trigraph, default).
            signed: If False (default), return max(difference, 0). If True, return the
                signed value.
            base: Logarithm base (default 2.0 for bits).

        Returns:
            DataFrame with columns: token_index, word, position, target,
            entropy_difference, available.
        """
        if n not in self.ENTROPY_DIFFERENCE_NS:
            raise ValueError(f"n must be one of {self.ENTROPY_DIFFERENCE_NS}, got {n!r}")
        df = self._assemble(tokens, signed=signed, base=base)
        out = df[
            [
                "token_index",
                "word",
                "position",
                "target",
                f"entropy_difference_{n}",
                f"entropy_difference_{n}_available",
            ]
        ].copy()
        out = out.rename(
            columns={
                f"entropy_difference_{n}": "entropy_difference",
                f"entropy_difference_{n}_available": "available",
            }
        )
        return out

    def compute_all(
        self, tokens: List[str], *, signed: bool = False, base: float = 2.0
    ) -> pd.DataFrame:
        """Compute all per-position character metrics in one pass.

        Convenience method for comparative analysis. See `surprisal`,
        `entropy_reduction`, and `entropy_difference` for the definitions.

        Args:
            tokens: List of token strings.
            signed: Applies to entropy_reduction / entropy_difference (default clipped).
            base: Logarithm base (default 2.0 for bits).

        Returns:
            DataFrame with one row per target character position: token_index, word,
            position, target, surprisal_{1,2,3}, entropy_reduction_{2,3},
            entropy_difference_{1,2,3}, and a matching ``*_available`` flag for each
            metric.
        """
        df = self._assemble(tokens, signed=signed, base=base)
        metric_cols = [f"surprisal_{n}" for n in self.SURPRISAL_NS]
        metric_cols += [f"entropy_reduction_{n}" for n in self.ENTROPY_REDUCTION_NS]
        metric_cols += [f"entropy_difference_{n}" for n in self.ENTROPY_DIFFERENCE_NS]
        flag_cols = [f"surprisal_{n}_available" for n in self.SURPRISAL_NS]
        flag_cols += [f"entropy_reduction_{n}_available" for n in self.ENTROPY_REDUCTION_NS]
        flag_cols += [f"entropy_difference_{n}_available" for n in self.ENTROPY_DIFFERENCE_NS]
        return df[["token_index", "word", "position", "target", *metric_cols, *flag_cols]].copy()

    # -------------------------------------------------------------------- lookups

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
