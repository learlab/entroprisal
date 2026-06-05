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
        min_frequency: Minimum frequency threshold for n-grams
    """

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
        self._build_gap_entropy_lookups()
        self._build_marginal_entropy()

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

    # Conditioning context lengths (number of preceding tokens), matching the n in
    # ngram_surprisal_n / ngram_entropy_n: n=1 is the bigram context, n=3 the 4-gram.
    # Surprisal supports n in {1, 2, 3}. Entropy reduction and entropy difference support
    # only n in {2, 3}. The n=1 cases are excluded as degenerate: entropy_reduction_1 =
    # H(W_t) - H(W_t | w_{t-1}) uses the scalar marginal as Distribution A, making it an
    # affine function of the mean transitional entropy, and the n=1 entropy difference
    # telescopes within a sequence to a length-confounded function of the final token's
    # entropy -- neither is a distinct construct.
    SURPRISAL_NS = (1, 2, 3)
    ENTROPY_REDUCTION_NS = (2, 3)
    ENTROPY_DIFFERENCE_NS = (2, 3)

    @staticmethod
    def _gap_context_cols(n: int) -> List[str]:
        """Distribution A context columns for context length n: the n-1 tokens up to w_{t-2}.

        These are the full n-token context (`_entropy_lookup[n]`) with the most recent
        token (w_{t-1} = token_2) dropped: n=3 -> [token_0, token_1]; n=2 -> [token_1].
        """
        return [f"token_{i}" for i in range(3 - n, 2)]

    def _build_gap_entropy_lookups(self):
        """Build the Distribution A entropy lookups for each entropy-reduction context length.

        For context length n, Distribution A conditions on the n-1 tokens ending at w_{t-2}
        and marginalizes over the dropped token w_{t-1} (and any earlier tokens not in the
        context) by summing counts. Summing counts across the dropped column *is* the
        marginalization, and is far cheaper than the weighted-sum formula given the
        count-table structure. The matching Distribution B is `_entropy_lookup[n]`.

        No smoothing is applied (raw MLE over the min_frequency-filtered table), so A and
        B are both unsmoothed. Entropies are stored in base 2; callers rescale for other
        bases.
        """
        self._gap_entropy_lookup = {}
        for n in self.ENTROPY_REDUCTION_NS:
            context_cols = self._gap_context_cols(n)

            if len(context_cols) == 1:
                key_expr = pl.col(context_cols[0]).cast(pl.Utf8).alias("context_key")
            else:
                key_expr = pl.concat_str(context_cols, separator="|||").alias("context_key")

            transitions = (
                self.ngram_df.select([*context_cols, "token_3", "count"])
                .filter(~pl.any_horizontal([pl.col(col) == "#" for col in context_cols]))
                .group_by([*context_cols, "token_3"])  # marginalize over dropped tokens
                .agg(pl.col("count").sum())
                .with_columns(key_expr)
            )

            context_totals = transitions.group_by("context_key").agg(
                pl.col("count").sum().alias("context_total")
            )

            self._gap_entropy_lookup[n] = (
                transitions.join(context_totals, on="context_key")
                .with_columns((pl.col("count") / pl.col("context_total")).alias("prob"))
                .group_by("context_key")
                .agg((-(pl.col("prob") * pl.col("prob").log(2)).sum()).alias("gap_entropy"))
            )

    def _build_marginal_entropy(self):
        """Compute the marginal token entropy H(W_t), used as Distribution A at n=1.

        This is the empty-context counterpart to `_gap_entropy_lookup`: summing counts
        across every preceding token yields the unconditional target distribution.
        `entropy_reduction_1 = H(W_t) - H(W_t | w_{t-1})` is the mutual information
        between adjacent tokens. Stored in base 2; callers rescale.
        """
        target_counts = (
            self.ngram_df.select(["token_3", "count"])
            .group_by("token_3")
            .agg(pl.col("count").sum())
        )
        total = float(target_counts["count"].sum())
        if total > 0:
            probs = (target_counts["count"].to_numpy().astype(float)) / total
            probs = probs[probs > 0]
            self.token_marginal_entropy = float(-(probs * np.log2(probs)).sum())
        else:
            self.token_marginal_entropy = 0.0

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
                - entropy_reduction_{2,3}: Mean conditional-mutual-information reduction
                  (clipped) over attested positions, by conditioning context length n
                  (n=3 is the 4-gram); see `entropy_reduction`
                - entropy_difference_{2,3}: Mean Lowder-style entropy difference
                  (clipped) over attested positions, by context length n; see
                  `entropy_difference`
                - entropy_reduction_{n}_support / entropy_difference_{n}_support: Number of
                  positions contributing to each mean

            Per-position values are available via `compute_all`, `surprisal`,
            `entropy_reduction`, and `entropy_difference`.
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

        # Per-position entropy metrics, averaged over attested positions (clipped, base 2)
        per_position = self._assemble(tokens, signed=False, base=2.0)
        metric_names = [f"entropy_reduction_{n}" for n in self.ENTROPY_REDUCTION_NS]
        metric_names += [f"entropy_difference_{n}" for n in self.ENTROPY_DIFFERENCE_NS]
        for metric_name in metric_names:
            values = per_position[metric_name].drop_nulls()
            metrics[f"{metric_name}_support"] = len(values)
            if len(values) > 0:
                metrics[metric_name] = float(values.mean())

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

    def _compute_per_position(self, tokens: List[str]) -> pl.DataFrame:
        """Build the raw per-position table (base-2, unclipped) with availability flags.

        Emits one row per token position. For each context length n, the next-word entropy
        ent_{n} = H(W_t | n preceding tokens) comes from `_entropy_lookup[n]`; this serves
        both as Distribution B for entropy reduction and as the term differenced across
        positions for entropy difference. gap_{n} is the matching Distribution A from
        `_gap_entropy_lookup[n]` (the n-1 tokens up to w_{t-2}). Surprisal is looked up at
        each context length (surprisal_{n} for n in SURPRISAL_NS); callers select the
        desired n. Unattested or out-of-range contexts produce nulls via left joins (no
        backoff).

        Returns one row per position holding (all base 2): position, token,
        surprisal_{n} for n in SURPRISAL_NS, ent_{n} for the entropy-reduction/difference
        context lengths, ent_prev_{n} (the previous position's ent_{n}, for entropy
        difference), and gap_{n} for n in ENTROPY_REDUCTION_NS.
        """
        ent_ns = sorted(set(self.ENTROPY_DIFFERENCE_NS) | set(self.ENTROPY_REDUCTION_NS))
        # bkey (full n-token context) is consumed by the surprisal joins as well as the
        # entropy joins, so build it for the union of those context lengths.
        bkey_ns = sorted(set(self.SURPRISAL_NS) | set(ent_ns))

        rows = []
        for t, tok in enumerate(tokens):
            row = {"position": t, "token": tok}
            # Full-context key for each context length n (n preceding tokens ending at t-1).
            for n in bkey_ns:
                row[f"bkey_{n}"] = "|||".join(tokens[t - n : t]) if t >= n else None
            # Distribution A (gappy) key: n-1 tokens ending at w_{t-2}.
            for n in self.ENTROPY_REDUCTION_NS:
                row[f"akey_{n}"] = "|||".join(tokens[t - n : t - 1]) if t >= n else None
            rows.append(row)

        schema = {"position": pl.Int64, "token": pl.Utf8}
        for n in bkey_ns:
            schema[f"bkey_{n}"] = pl.Utf8
        for n in self.ENTROPY_REDUCTION_NS:
            schema[f"akey_{n}"] = pl.Utf8

        result = pl.DataFrame(rows, schema=schema)

        # Surprisal at each context length: surprisal_{n} = -log2 P(w_t | n preceding tokens).
        for n in self.SURPRISAL_NS:
            result = result.join(
                self._surprisal_lookup[n].rename({"surprisal": f"surprisal_{n}"}),
                left_on=[f"bkey_{n}", "token"],
                right_on=["context_key", "target"],
                how="left",
            )

        # Next-word entropy ent_{n} for each context length (Distribution B / difference term).
        for n in ent_ns:
            result = result.join(
                self._entropy_lookup[n].rename({"entropy": f"ent_{n}"}),
                left_on=f"bkey_{n}",
                right_on="context_key",
                how="left",
            )

        # Distribution A (gappy) entropy for each entropy-reduction context length.
        for n in self.ENTROPY_REDUCTION_NS:
            result = result.join(
                self._gap_entropy_lookup[n].rename({"gap_entropy": f"gap_{n}"}),
                left_on=f"akey_{n}",
                right_on="context_key",
                how="left",
            )

        # Previous position's next-word entropy at each length, for entropy_difference.
        result = result.sort("position").with_columns(
            [pl.col(f"ent_{n}").shift(1).alias(f"ent_prev_{n}") for n in ent_ns]
        )

        return result

    @staticmethod
    def _base_factor(base: float) -> float:
        """Conversion factor from base-2 information units to the requested base."""
        if base == 2.0:
            return 1.0
        return float(np.log(2.0) / np.log(base))

    @staticmethod
    def _clip_nonnegative(column: str) -> "pl.Expr":
        """max(column, 0) while preserving nulls (null < 0 is null -> falls through)."""
        return pl.when(pl.col(column) < 0).then(0.0).otherwise(pl.col(column)).alias(column)

    def _assemble(self, tokens: List[str], signed: bool, base: float) -> pl.DataFrame:
        """Compute scaled, optionally-clipped per-position metrics for all context lengths.

        entropy_reduction_{n} = H(A_n) - H(B_n)  (conditional mutual information form).
        entropy_difference_{n} = E_n[t-1] - E_n[t]  (Lowder-style, distinct random vars).
        Suffix n is the conditioning context length, matching ngram_surprisal_n (n=3 is the
        4-gram). All default to max(., 0); pass signed=True to keep negative values.
        Availability flags are derived per metric (True only when every context matched).
        """
        df = self._compute_per_position(tokens)
        factor = self._base_factor(base)

        derived = []
        flags = []
        clip_targets = []

        for n in self.SURPRISAL_NS:
            derived.append((pl.col(f"surprisal_{n}") * factor).alias(f"surprisal_{n}"))
            flags.append(pl.col(f"surprisal_{n}").is_not_null().alias(f"surprisal_{n}_available"))

        for n in self.ENTROPY_REDUCTION_NS:
            derived.append(
                ((pl.col(f"gap_{n}") - pl.col(f"ent_{n}")) * factor).alias(
                    f"entropy_reduction_{n}"
                )
            )
            flags.append(
                (pl.col(f"gap_{n}").is_not_null() & pl.col(f"ent_{n}").is_not_null()).alias(
                    f"entropy_reduction_{n}_available"
                )
            )
            clip_targets.append(f"entropy_reduction_{n}")

        for n in self.ENTROPY_DIFFERENCE_NS:
            derived.append(
                ((pl.col(f"ent_prev_{n}") - pl.col(f"ent_{n}")) * factor).alias(
                    f"entropy_difference_{n}"
                )
            )
            flags.append(
                (pl.col(f"ent_{n}").is_not_null() & pl.col(f"ent_prev_{n}").is_not_null()).alias(
                    f"entropy_difference_{n}_available"
                )
            )
            clip_targets.append(f"entropy_difference_{n}")

        df = df.with_columns(derived + flags)

        if not signed:
            df = df.with_columns([self._clip_nonnegative(col) for col in clip_targets])

        return df

    def surprisal(self, tokens: List[str], *, n: int = 3, base: float = 2.0) -> pd.DataFrame:
        """Per-position surprisal: -log P(w_t | n preceding tokens).

        High when the observed word was unlikely given its preceding context. No backoff
        is applied: positions whose context is unattested (or which lack n preceding
        tokens) get NA and surprisal_available == False.

        - n=3 (default, 4-gram): -log P(w_t | w_{t-3}, w_{t-2}, w_{t-1}).
        - n=2 (trigram):         -log P(w_t | w_{t-2}, w_{t-1}).
        - n=1 (bigram):          -log P(w_t | w_{t-1}).

        Args:
            tokens: List of token strings (e.g. from preprocess_text(text)[0]).
            n: Conditioning context length, 1, 2, or 3 (default 3).
            base: Logarithm base (default 2.0 for bits).

        Returns:
            DataFrame with columns: position, token, surprisal, surprisal_available.
        """
        if n not in self.SURPRISAL_NS:
            raise ValueError(f"n must be one of {self.SURPRISAL_NS}, got {n!r}")
        df = self._assemble(tokens, signed=True, base=base)
        return df.select(
            [
                "position",
                "token",
                pl.col(f"surprisal_{n}").alias("surprisal"),
                pl.col(f"surprisal_{n}_available").alias("surprisal_available"),
            ]
        ).to_pandas()

    def entropy_reduction(
        self, tokens: List[str], *, n: int = 3, signed: bool = False, base: float = 2.0
    ) -> pd.DataFrame:
        """Per-position entropy reduction (conditional mutual information form).

        For conditioning context length ``n`` (matching ngram_surprisal_n), computes
        H(W_t | w_{t-n}..w_{t-2}) - H(W_t | w_{t-n}..w_{t-1}): how much observing the most
        recent context word w_{t-1} reduced uncertainty about the fixed target w_t.

        - n=3 (default, the 4-gram): H(W_t | w_{t-3}, w_{t-2}) - H(W_t | w_{t-3..t-1}).
        - n=2 (the trigram):         H(W_t | w_{t-2})         - H(W_t | w_{t-2}, w_{t-1}).

        n=1 is unsupported: H(W_t) - H(W_t | w_{t-1}) uses the scalar marginal as
        Distribution A, making the per-sequence mean an affine function of the mean
        transitional entropy rather than a distinct construct.

        This is the pointwise conditional mutual information I(W_{t-1}; W_t | earlier
        context) and has the clean H(X) - H(X|y) form Hale (2016) describes, restricted to
        the next word. Distribution A (before w_{t-1}) pools across more continuations than
        Distribution B, so it has wider support; this is correct, not a bug. When the
        continuation is deterministic (P(w_{t-1} | earlier context) = 1), A == B and the
        reduction is exactly 0. No smoothing is applied.

        Args:
            tokens: List of token strings.
            n: Conditioning context length, 2 (trigram) or 3 (4-gram, default). Drops the
                most recent context token and marginalizes over it. n=1 is unsupported
                (it reduces to an affine function of the mean transitional entropy).
            signed: If False (default), return max(reduction, 0) per Hale's convention.
                If True, return the signed value (negative means the new context
                broadened the expectation space).
            base: Logarithm base (default 2.0 for bits).

        Returns:
            DataFrame with columns: position, token, entropy_reduction, available. NA
            where either contributing context is unattested.

        See also `entropy_difference` for the Lowder-style version.
        """
        if n not in self.ENTROPY_REDUCTION_NS:
            raise ValueError(f"n must be one of {self.ENTROPY_REDUCTION_NS}, got {n!r}")
        df = self._assemble(tokens, signed=signed, base=base)
        return df.select(
            [
                "position",
                "token",
                pl.col(f"entropy_reduction_{n}").alias("entropy_reduction"),
                pl.col(f"entropy_reduction_{n}_available").alias("available"),
            ]
        ).to_pandas()

    def entropy_difference(
        self, tokens: List[str], *, n: int = 3, signed: bool = False, base: float = 2.0
    ) -> pd.DataFrame:
        """Per-position entropy difference (Lowder-style), clipped at zero by default.

        For conditioning context length ``n``, computes E_n[t-1] - E_n[t], where
        E_n[t] = H(W_t | w_{t-n}..w_{t-1}) is the next-word entropy at position t. n=3
        (the 4-gram) reproduces the original Lowder definition
        H(W_{t-1} | w_{t-4..t-2}) - H(W_t | w_{t-3..t-1}); n=2 uses the trigram context.

        This corresponds to the quantity Lowder et al. (2018) termed "entropy reduction".
        Note that it differences entropies over distinct random variables (guesses at
        adjacent positions) rather than computing H(X) - H(X|y) over a fixed target. It
        is a descriptive statistic capturing whether the next-word distribution has
        become more peaked from one position to the next, not an information-theoretic
        reduction in the Shannon sense. See `entropy_reduction` for the
        conditional-mutual-information version.

        Args:
            tokens: List of token strings.
            n: Conditioning context length, 2 (trigram) or 3 (4-gram, default).
                n=1 is unsupported: its within-sequence differences telescope to a
                function of the final token's entropy (a length-confounded restatement).
            signed: If False (default), return max(difference, 0). If True, return the
                signed value.
            base: Logarithm base (default 2.0 for bits).

        Returns:
            DataFrame with columns: position, token, entropy_difference, available. NA
            where either position's entropy is unattested (including the first scorable
            position, which has no predecessor).
        """
        if n not in self.ENTROPY_DIFFERENCE_NS:
            raise ValueError(f"n must be one of {self.ENTROPY_DIFFERENCE_NS}, got {n!r}")
        df = self._assemble(tokens, signed=signed, base=base)
        return df.select(
            [
                "position",
                "token",
                pl.col(f"entropy_difference_{n}").alias("entropy_difference"),
                pl.col(f"entropy_difference_{n}_available").alias("available"),
            ]
        ).to_pandas()

    def compute_all(
        self, tokens: List[str], *, signed: bool = False, base: float = 2.0
    ) -> pd.DataFrame:
        """Compute all per-position metrics, at every context length, in one pass.

        Convenience method for comparative analysis. See `surprisal`, `entropy_reduction`,
        and `entropy_difference` for the definitions. Suffix n is the conditioning context
        length, matching ngram_surprisal_n (n=3 is the 4-gram).

        Args:
            tokens: List of token strings.
            signed: Applies to entropy_reduction / entropy_difference (default clipped).
            base: Logarithm base (default 2.0 for bits).

        Returns:
            DataFrame with one row per token position: position, token,
            surprisal_{1,2,3}, entropy_reduction_{2,3}, entropy_difference_{2,3}, and a
            matching ``*_available`` flag for each metric.
        """
        df = self._assemble(tokens, signed=signed, base=base)
        metric_cols = [f"surprisal_{n}" for n in self.SURPRISAL_NS]
        metric_cols += [f"entropy_reduction_{n}" for n in self.ENTROPY_REDUCTION_NS]
        metric_cols += [f"entropy_difference_{n}" for n in self.ENTROPY_DIFFERENCE_NS]
        flag_cols = [f"surprisal_{n}_available" for n in self.SURPRISAL_NS]
        flag_cols += [f"entropy_reduction_{n}_available" for n in self.ENTROPY_REDUCTION_NS]
        flag_cols += [f"entropy_difference_{n}_available" for n in self.ENTROPY_DIFFERENCE_NS]
        return df.select(["position", "token", *metric_cols, *flag_cols]).to_pandas()

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
