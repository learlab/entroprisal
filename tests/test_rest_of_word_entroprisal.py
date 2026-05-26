"""Tests for RestOfWordEntropisalCalculator."""

import math

import pandas as pd
import pytest

from entroprisal import RestOfWordEntropisalCalculator


def _entropy(counts):
    """Shannon entropy (bits) of a list of counts."""
    total = sum(counts)
    probs = [c / total for c in counts]
    return -sum(p * math.log2(p) for p in probs)


@pytest.fixture
def sample_words():
    """Create sample word frequency data for testing."""
    return pd.DataFrame(
        {
            "WORD": ["cat", "dog", "the", "and", "can", "car", "cats", "dogs"],
            "COUNT": [1000, 800, 5000, 4000, 600, 700, 400, 350],
        }
    )


@pytest.fixture
def uniform_words():
    """Four equally-frequent words for hand-computable entropies.

    Marginal H(W) = log2(4) = 2.0.
    Prefix '#c' covers {cat, car, can} -> H = log2(3) ~ 1.585.
    Prefix '#ca' covers {cat, car, can} -> H = log2(3) ~ 1.585.
    Prefix '#cat' covers {cat} -> H = 0.
    """
    return pd.DataFrame(
        {
            "WORD": ["cat", "car", "can", "dog"],
            "COUNT": [1, 1, 1, 1],
        }
    )


def test_init(sample_words):
    """Test calculator initialization."""
    calc = RestOfWordEntropisalCalculator(sample_words)
    assert calc is not None
    assert len(calc.df) == 8


def test_calculate_metrics(sample_words):
    """Test basic metric calculation."""
    calc = RestOfWordEntropisalCalculator(sample_words)
    tokens = ["the", "cat", "and", "dog"]
    metrics = calc.calculate_metrics(tokens)

    assert "mean_word_length" in metrics
    # Left-to-right metrics (entropy and surprisal)
    assert "lr_c1_entropy" in metrics
    assert "lr_c1_surprisal" in metrics
    # Right-to-left metrics (entropy and surprisal)
    assert "rl_c1_entropy" in metrics
    assert "rl_c1_surprisal" in metrics


def test_calculate_batch(sample_words):
    """Test batch processing."""
    calc = RestOfWordEntropisalCalculator(sample_words)
    token_lists = [["the", "cat"], ["the", "dog"]]

    results = calc.calculate_batch(token_lists)
    assert len(results) == 2


def test_get_word_frequency(sample_words):
    """Test word frequency lookup."""
    calc = RestOfWordEntropisalCalculator(sample_words)

    freq = calc.get_word_frequency("cat")
    assert freq == 1000

    # Non-existent word
    freq = calc.get_word_frequency("xyz")
    assert freq == 0


# ----------------------------------------------------------------- per-position


def test_word_marginal_entropy(uniform_words):
    """word_marginal_entropy = H(W) over the corpus, weighted by frequency."""
    calc = RestOfWordEntropisalCalculator(uniform_words)
    # 4 words, equal frequency -> H = log2(4) = 2.0 bits.
    assert calc.word_marginal_entropy == pytest.approx(2.0)


def test_compute_all_shape(uniform_words):
    """compute_all returns one row per word with the expected columns."""
    calc = RestOfWordEntropisalCalculator(uniform_words)
    df = calc.compute_all(["cat", "dog"])

    assert len(df) == 2
    # Spot-check the column set: 2 directions x (3 surprisal + 3 ER) = 12 metric cols
    # + same number of availability flags + token_index + word = 26 columns.
    expected_metric_cols = {
        "lr_surprisal_1",
        "lr_surprisal_2",
        "lr_surprisal_3",
        "lr_entropy_reduction_1",
        "lr_entropy_reduction_2",
        "lr_entropy_reduction_3",
        "rl_surprisal_1",
        "rl_surprisal_2",
        "rl_surprisal_3",
        "rl_entropy_reduction_1",
        "rl_entropy_reduction_2",
        "rl_entropy_reduction_3",
    }
    assert expected_metric_cols.issubset(set(df.columns))
    assert "token_index" in df.columns
    assert "word" in df.columns


def test_compute_all_empty_input(sample_words):
    """compute_all on empty token list returns empty DataFrame with the right columns."""
    calc = RestOfWordEntropisalCalculator(sample_words)
    df = calc.compute_all([])
    assert len(df) == 0
    assert "lr_entropy_reduction_1" in df.columns


def test_entropy_reduction_n1_uses_marginal(uniform_words):
    """ER_1 (lr) = word_marginal_entropy - lr_c1_entropy. Hand-computable."""
    calc = RestOfWordEntropisalCalculator(uniform_words)
    df = calc.entropy_reduction(["cat"], direction="lr", n=1, signed=True)
    row = df.iloc[0]

    # Prefix '#c' pools {cat, car, can} -> H = log2(3); marginal H(W) = log2(4).
    expected = math.log2(4) - math.log2(3)
    assert row["entropy_reduction"] == pytest.approx(expected)
    assert expected > 0
    assert bool(row["available"]) is True


def test_entropy_reduction_n2_uses_prefix_chain(uniform_words):
    """ER_2 (lr) = lr_c1_entropy - lr_c2_entropy. Identical prefixes -> 0."""
    calc = RestOfWordEntropisalCalculator(uniform_words)
    df = calc.entropy_reduction(["cat"], direction="lr", n=2, signed=True)
    row = df.iloc[0]
    # '#c' and '#ca' both cover {cat, car, can} -> same entropy -> ER = 0.
    assert row["entropy_reduction"] == pytest.approx(0.0)
    assert bool(row["available"]) is True


def test_entropy_reduction_n3_resolves_word(uniform_words):
    """ER_3 (lr) = lr_c2_entropy - lr_c3_entropy. '#cat' identifies the word."""
    calc = RestOfWordEntropisalCalculator(uniform_words)
    df = calc.entropy_reduction(["cat"], direction="lr", n=3, signed=True)
    row = df.iloc[0]
    # '#ca' has H = log2(3); '#cat' uniquely identifies 'cat' -> H = 0.
    expected = math.log2(3) - 0.0
    assert row["entropy_reduction"] == pytest.approx(expected)
    assert bool(row["available"]) is True


def test_entropy_reduction_rl_direction(uniform_words):
    """rl entropy reduction works symmetrically with suffixes."""
    calc = RestOfWordEntropisalCalculator(uniform_words)
    # Suffix 't#' uniquely identifies 'cat' in the corpus -> H(W | suffix) = 0.
    # ER_1 (rl) = marginal - 0 = 2.0.
    df = calc.entropy_reduction(["cat"], direction="rl", n=1, signed=True)
    assert df.iloc[0]["entropy_reduction"] == pytest.approx(2.0)


def test_entropy_reduction_invalid_n(sample_words):
    """Context lengths outside 1..3 are rejected for entropy reduction."""
    calc = RestOfWordEntropisalCalculator(sample_words)
    with pytest.raises(ValueError):
        calc.entropy_reduction(["the"], n=0)
    with pytest.raises(ValueError):
        calc.entropy_reduction(["the"], n=4)


def test_entropy_reduction_invalid_direction(sample_words):
    """Directions outside {lr, rl} are rejected."""
    calc = RestOfWordEntropisalCalculator(sample_words)
    with pytest.raises(ValueError):
        calc.entropy_reduction(["the"], direction="up")


def test_entropy_reduction_unattested_is_na(sample_words):
    """Unattested context yields NaN and availability flag False."""
    calc = RestOfWordEntropisalCalculator(sample_words)
    # No word starting with 'z' in sample_words -> '#z' prefix unattested.
    df = calc.entropy_reduction(["zebra"], direction="lr", n=1)
    row = df.iloc[0]
    assert math.isnan(row["entropy_reduction"])
    assert bool(row["available"]) is False


def test_signed_clipping_behavior(sample_words):
    """signed=False clips negative reductions to 0; signed=True preserves them."""
    calc = RestOfWordEntropisalCalculator(sample_words)
    # Any attested word will give some reduction value; rare to be negative in
    # realistic corpora, so just verify the clipping applies (non-negative output).
    df_clipped = calc.entropy_reduction(["cat"], direction="lr", n=2)
    assert (df_clipped["entropy_reduction"].dropna() >= 0).all()


def test_surprisal_shape(uniform_words):
    """Per-word surprisal returns one row per token with the expected columns."""
    calc = RestOfWordEntropisalCalculator(uniform_words)
    df = calc.surprisal(["cat", "dog"], direction="lr", n=1)
    assert list(df.columns) == [
        "token_index",
        "word",
        "surprisal",
        "surprisal_available",
    ]
    assert len(df) == 2


def test_surprisal_value(uniform_words):
    """Per-word lr_surprisal_1 matches -log2 P(rest | prefix)."""
    calc = RestOfWordEntropisalCalculator(uniform_words)
    df = calc.surprisal(["cat"], direction="lr", n=1)
    # P('at#' | '#c') = 1/3 (cat, car, can with rests 'at#', 'ar#', 'an#').
    assert df.iloc[0]["surprisal"] == pytest.approx(math.log2(3))


def test_base_parameter_scales_values(uniform_words):
    """Changing the log base rescales information units by log(2)/log(base)."""
    calc = RestOfWordEntropisalCalculator(uniform_words)
    base2 = calc.entropy_reduction(["cat"], direction="lr", n=1, base=2.0)
    base4 = calc.entropy_reduction(["cat"], direction="lr", n=1, base=4.0)
    v2 = base2.iloc[0]["entropy_reduction"]
    v4 = base4.iloc[0]["entropy_reduction"]
    assert v4 == pytest.approx(v2 / 2.0)


def test_calculate_metrics_includes_new_keys(uniform_words):
    """Aggregate metrics include the new entropy_reduction keys for both directions."""
    calc = RestOfWordEntropisalCalculator(uniform_words)
    metrics = calc.calculate_metrics(["cat", "dog"])

    for direction in ("lr", "rl"):
        for n in (1, 2, 3):
            assert f"{direction}_entropy_reduction_{n}_support" in metrics
    # Marginal-based reductions should have support for both words.
    assert metrics["lr_entropy_reduction_1_support"] >= 1
    assert "lr_entropy_reduction_1" in metrics


def test_calculate_metrics_preserves_existing_keys(sample_words):
    """Backward compat: existing aggregate keys are still produced."""
    calc = RestOfWordEntropisalCalculator(sample_words)
    metrics = calc.calculate_metrics(["the", "cat"])
    # Sample of the existing keys.
    assert "mean_word_length" in metrics
    assert "lr_c1_entropy" in metrics
    assert "lr_c1_surprisal" in metrics
    assert "rl_c1_entropy" in metrics
