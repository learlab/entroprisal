"""Tests for TokenEntropisalCalculator."""

import math

import polars as pl
import pytest

from entroprisal import TokenEntropisalCalculator


def _entropy(counts):
    """Shannon entropy (bits) of a list of counts."""
    total = sum(counts)
    probs = [c / total for c in counts]
    return -sum(p * math.log2(p) for p in probs)


@pytest.fixture
def sample_ngrams():
    """Create sample n-gram data for testing."""
    return pl.LazyFrame(
        {
            "token_0": ["the", "the", "the", "cat", "cat", "sat"],
            "token_1": ["cat", "cat", "dog", "sat", "sat", "on"],
            "token_2": ["sat", "ran", "ran", "on", "down", "the"],
            "token_3": ["down", "away", "fast", "the", "slowly", "mat"],
            "count": [100, 50, 30, 80, 20, 60],
        }
    )


def test_init(sample_ngrams):
    """Test calculator initialization."""
    calc = TokenEntropisalCalculator(sample_ngrams, min_frequency=10)
    assert calc is not None
    assert calc.min_frequency == 10


def test_calculate_metrics(sample_ngrams):
    """Test basic metric calculation."""
    calc = TokenEntropisalCalculator(sample_ngrams, min_frequency=10)
    metrics = calc.calculate_metrics(["the", "cat", "sat"])

    assert "n_tokens" in metrics
    assert metrics["n_tokens"] == 3
    assert "mean_token_length" in metrics


def test_calculate_batch(sample_ngrams):
    """Test batch processing."""
    calc = TokenEntropisalCalculator(sample_ngrams, min_frequency=10)
    token_lists = [
        ["the", "cat"],
        ["the", "dog"],
    ]

    results = calc.calculate_batch(token_lists)
    assert len(results) == 2
    assert "n_tokens" in results.columns


@pytest.fixture
def rich_ngrams():
    """N-gram data with enough structure to exercise the per-position metrics.

    Contexts of interest:
      - (the, cat, sat) -> {down:100, up:100}        non-trivial surprisal / entropy
      - (the, cat)      -> {down:100, up:100, away:50} via marginalizing token_2
      - (foo, bar, baz) -> {qux:40, quux:60}; (foo, bar) collapses to the same dist
        (deterministic token_2) -> entropy_reduction == 0
      - (cat, sat, down) -> {hard:70}                chains after (the, cat, sat)
      - (p, q, *)        peaked when pooled, flat for token_2 = n -> negative reduction
    """
    return pl.LazyFrame(
        {
            "token_0": ["the", "the", "the", "the", "foo", "foo", "cat", "p", "p", "p", "p"],
            "token_1": ["cat", "cat", "cat", "dog", "bar", "bar", "sat", "q", "q", "q", "q"],
            "token_2": ["sat", "sat", "ran", "ran", "baz", "baz", "down", "m", "n", "n", "n"],
            "token_3": ["down", "up", "away", "fast", "qux", "quux", "hard", "x", "x", "y", "z"],
            "count": [100, 100, 50, 30, 40, 60, 70, 900, 10, 10, 10],
        }
    )


def test_compute_all_shape(rich_ngrams):
    """compute_all returns one row per token position with the expected columns."""
    calc = TokenEntropisalCalculator(rich_ngrams, min_frequency=10)
    tokens = ["the", "cat", "sat", "down"]
    df = calc.compute_all(tokens)

    assert len(df) == len(tokens)
    assert list(df.columns) == [
        "position",
        "token",
        "surprisal",
        "entropy_reduction_2",
        "entropy_reduction_3",
        "entropy_difference_1",
        "entropy_difference_2",
        "entropy_difference_3",
        "surprisal_available",
        "entropy_reduction_2_available",
        "entropy_reduction_3_available",
        "entropy_difference_1_available",
        "entropy_difference_2_available",
        "entropy_difference_3_available",
    ]


def test_per_position_surprisal_value(rich_ngrams):
    """Surprisal matches -log2(count / context_total) and flags boundary positions."""
    calc = TokenEntropisalCalculator(rich_ngrams, min_frequency=10)
    df = calc.surprisal(["the", "cat", "sat", "down"])

    scored = df[df["position"] == 3].iloc[0]
    # P(down | the, cat, sat) = 100 / 200 -> 1 bit
    assert scored["surprisal"] == pytest.approx(1.0)
    assert bool(scored["surprisal_available"]) is True

    # Positions lacking a full trigram context are NA + flag False, not dropped.
    for pos in (0, 1, 2):
        row = df[df["position"] == pos].iloc[0]
        assert bool(row["surprisal_available"]) is False
        assert math.isnan(row["surprisal"])


def test_entropy_reduction_positive(rich_ngrams):
    """H(A) - H(B) is positive when observing w_{t-1} sharpens the distribution (n=3)."""
    calc = TokenEntropisalCalculator(rich_ngrams, min_frequency=10)
    df = calc.entropy_reduction(["the", "cat", "sat", "down"])  # default n=3 (4-gram)
    row = df[df["position"] == 3].iloc[0]

    # A = H(the, cat) marginalizing token_2 = {down,up,away}; B = H(the, cat, sat).
    expected = _entropy([100, 100, 50]) - _entropy([100, 100])
    assert row["entropy_reduction"] == pytest.approx(expected)
    assert expected > 0
    assert bool(row["available"]) is True


def test_entropy_reduction_n2(rich_ngrams):
    """n=2 (trigram) reduction: H(W_t | w_{t-2}) - H(W_t | w_{t-2}, w_{t-1})."""
    calc = TokenEntropisalCalculator(rich_ngrams, min_frequency=10)
    # tokens[t-2]=cat (w_{t-2}), tokens[t-1]=sat (w_{t-1}), target=down at position 2.
    df = calc.entropy_reduction(["cat", "sat", "down"], n=2)
    row = df[df["position"] == 2].iloc[0]

    # A = H(cat) over token_3 = {down:100, up:100, away:50}; B = H(cat, sat) = {down,up}.
    expected = _entropy([100, 100, 50]) - _entropy([100, 100])
    assert row["entropy_reduction"] == pytest.approx(expected)
    assert expected > 0
    assert bool(row["available"]) is True


def test_entropy_reduction_invalid_n(rich_ngrams):
    """Context lengths other than 2 or 3 are rejected for entropy reduction."""
    calc = TokenEntropisalCalculator(rich_ngrams, min_frequency=10)
    with pytest.raises(ValueError):
        calc.entropy_reduction(["the", "cat", "sat", "down"], n=1)


def test_entropy_reduction_deterministic_is_zero(rich_ngrams):
    """When the dropped context token is deterministic, A == B so reduction == 0."""
    calc = TokenEntropisalCalculator(rich_ngrams, min_frequency=10)

    # n=3 (4-gram): (foo, bar) -> only token_2 = baz.
    row3 = calc.entropy_reduction(["foo", "bar", "baz", "qux"]).iloc[-1]
    assert row3["entropy_reduction"] == pytest.approx(0.0, abs=1e-12)

    # n=2 (trigram): (bar) -> only token_2 = baz.
    row2 = calc.entropy_reduction(["bar", "baz", "qux"], n=2).iloc[-1]
    assert row2["entropy_reduction"] == pytest.approx(0.0, abs=1e-12)


def test_entropy_reduction_signed_can_be_negative(rich_ngrams):
    """Signed reduction is negative when the new context broadens expectations; clipped -> 0."""
    calc = TokenEntropisalCalculator(rich_ngrams, min_frequency=10)
    tokens = ["p", "q", "n", "x"]

    signed = calc.entropy_reduction(tokens, signed=True)
    signed_row = signed[signed["position"] == 3].iloc[0]
    expected = _entropy([910, 10, 10]) - _entropy([10, 10, 10])
    assert signed_row["entropy_reduction"] == pytest.approx(expected)
    assert expected < 0

    clipped = calc.entropy_reduction(tokens)  # default clipped
    clipped_row = clipped[clipped["position"] == 3].iloc[0]
    assert clipped_row["entropy_reduction"] == pytest.approx(0.0)


def test_entropy_reduction_unattested_is_na(rich_ngrams):
    """Unattested contexts yield NA and availability flags set to False (no backoff)."""
    calc = TokenEntropisalCalculator(rich_ngrams, min_frequency=10)
    df = calc.entropy_reduction(["zzz", "yyy", "xxx", "www"])
    row = df[df["position"] == 3].iloc[0]

    assert math.isnan(row["entropy_reduction"])
    assert bool(row["available"]) is False


def test_entropy_difference_shift_and_clip(rich_ngrams):
    """entropy_difference = E[t-1] - E[t]; first scorable position has no predecessor (n=3)."""
    calc = TokenEntropisalCalculator(rich_ngrams, min_frequency=10)
    df = calc.entropy_difference(["the", "cat", "sat", "down", "hard"])  # default n=3

    # First scorable position (t=3): no predecessor -> NA.
    first = df[df["position"] == 3].iloc[0]
    assert math.isnan(first["entropy_difference"])
    assert bool(first["available"]) is False

    # t=4: E[t-1] = H(the,cat,sat) = 1 bit, E[t] = H(cat,sat,down) = 0 -> diff = 1.0.
    second = df[df["position"] == 4].iloc[0]
    assert second["entropy_difference"] == pytest.approx(1.0)
    assert bool(second["available"]) is True


def test_entropy_difference_lower_order(rich_ngrams):
    """n=1 (bigram) entropy difference: E_1[t-1] - E_1[t] over single-token contexts."""
    calc = TokenEntropisalCalculator(rich_ngrams, min_frequency=10)
    df = calc.entropy_difference(["sat", "down", "hard"], n=1)

    # E_1[down] context "sat" = H({down:100, up:100}) = 1.0 bit;
    # E_1[hard] context "down" = H({hard:70}) = 0 -> diff at t=2 is 1.0.
    row = df[df["position"] == 2].iloc[0]
    assert row["entropy_difference"] == pytest.approx(1.0)
    assert bool(row["available"]) is True


def test_entropy_difference_invalid_n(rich_ngrams):
    """Context lengths outside 1..3 are rejected for entropy difference."""
    calc = TokenEntropisalCalculator(rich_ngrams, min_frequency=10)
    with pytest.raises(ValueError):
        calc.entropy_difference(["the", "cat", "sat", "down"], n=4)


def test_base_parameter_scales_values(rich_ngrams):
    """Changing the log base rescales information units by log(2)/log(base)."""
    calc = TokenEntropisalCalculator(rich_ngrams, min_frequency=10)
    base2 = calc.surprisal(["the", "cat", "sat", "down"])
    base4 = calc.surprisal(["the", "cat", "sat", "down"], base=4.0)

    s2 = base2[base2["position"] == 3].iloc[0]["surprisal"]
    s4 = base4[base4["position"] == 3].iloc[0]["surprisal"]
    assert s2 == pytest.approx(1.0)
    assert s4 == pytest.approx(0.5)  # 1 bit -> 0.5 in base-4 units


def test_calculate_metrics_includes_entropy_keys(rich_ngrams):
    """Aggregate metrics fold in mean entropy_reduction / entropy_difference + support."""
    calc = TokenEntropisalCalculator(rich_ngrams, min_frequency=10)
    metrics = calc.calculate_metrics(["the", "cat", "sat", "down", "hard"])

    assert "entropy_reduction_2_support" in metrics
    assert "entropy_reduction_3_support" in metrics
    assert "entropy_difference_1_support" in metrics
    assert "entropy_difference_3_support" in metrics
    assert metrics["entropy_reduction_3_support"] >= 1
    assert "entropy_reduction_3" in metrics
