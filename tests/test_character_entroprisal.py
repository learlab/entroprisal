"""Tests for CharacterEntropisalCalculator."""

import math

import pandas as pd
import pytest

from entroprisal import CharacterEntropisalCalculator


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
            "WORD": ["cat", "dog", "the", "and", "can", "car"],
            "COUNT": [1000, 800, 5000, 4000, 600, 700],
        }
    )


@pytest.fixture
def crafted_words():
    """Crafted word frequencies with hand-computable per-position metrics.

    For target at boundary-padded position 2 across {ab, ac, ba, bc}:
      - gap_{n=2} context '#': pooled second-char distribution {b:1, c:2, a:1}
        -> H = 1.5 bits.
      - bigraph context '#a': second-char distribution {b:1, c:1} -> H = 1.0 bit.
      - entropy_reduction_2 = 1.5 - 1.0 = 0.5 bit  (positive).

    For target at position 2 in {de, df} (always 'd' after '#'):
      - gap context '#' pools {b:1, c:2, a:1, e:1, f:1} -> H ~ 2.058 bits
      - bigraph context '#d': second-char distribution {e:1, f:1} -> H = 1.0 bit
      - but importantly the dropped char word[1] is deterministic given word[0]='#'
        when restricted to words starting with 'd'. For the deterministic test below
        we use a separate fixture.
    """
    return pd.DataFrame(
        {
            "WORD": ["ab", "ac", "ba", "bc"],
            "COUNT": [1, 1, 1, 1],
        }
    )


@pytest.fixture
def deterministic_words():
    """Words where word[1] is deterministic given word[0]='#': all start with 'd'.

    For any target at position 2:
      - gap context '#' (marginalize over word[1] which is always 'd'):
        target distribution is the same as bigraph '#d' distribution.
      - Therefore entropy_reduction_2 == 0 exactly.
    """
    return pd.DataFrame(
        {
            "WORD": ["de", "df", "dg"],
            "COUNT": [1, 1, 1],
        }
    )


@pytest.fixture
def peaked_words():
    """Pooled marginal is peaked but a specific full context is uniform.

    Words and counts chosen so the gap_2 distribution at '#' is dominated by one
    second-char (entropy near zero), while the bigraph '#p' distribution is uniform
    (high entropy). Then signed entropy_reduction_2 at a '#p_' target is negative.
    """
    return pd.DataFrame(
        {
            "WORD": ["qa", "qa", "qa", "qa", "qa", "qa", "qa", "qa", "qa", "pa", "pb"],
            "COUNT": [10, 10, 10, 10, 10, 10, 10, 10, 10, 1, 1],
        }
    )


def test_init(sample_words):
    """Test calculator initialization."""
    calc = CharacterEntropisalCalculator(sample_words)
    assert calc is not None
    assert len(calc.df) == 6


def test_calculate_metrics(sample_words):
    """Test basic metric calculation."""
    calc = CharacterEntropisalCalculator(sample_words)
    tokens = ["the", "cat", "and", "dog"]
    metrics = calc.calculate_metrics(tokens)

    assert "char_entropy" in metrics
    assert "char_surprisal" in metrics
    assert "bigraph_entropy" in metrics
    assert "bigraph_surprisal" in metrics
    assert "trigraph_entropy" in metrics
    assert "trigraph_surprisal" in metrics


def test_calculate_batch(sample_words):
    """Test batch processing."""
    calc = CharacterEntropisalCalculator(sample_words)
    token_lists = [["the", "cat"], ["the", "dog"]]

    results = calc.calculate_batch(token_lists)
    assert len(results) == 2


def test_get_character_entropy(sample_words):
    """Test character entropy lookup."""
    calc = CharacterEntropisalCalculator(sample_words)

    # Should return a value for common character
    entropy = calc.get_character_entropy("c")
    assert entropy is not None
    assert entropy >= 0


def test_get_character_surprisal(sample_words):
    """Test character surprisal lookup."""
    calc = CharacterEntropisalCalculator(sample_words)

    # Test surprisal lookup
    surprisal = calc.get_character_surprisal("c", "a")
    assert surprisal is None or surprisal >= 0


# ----------------------------------------------------------------- per-position


def test_compute_all_shape(crafted_words):
    """compute_all returns one row per target char position with the expected columns."""
    calc = CharacterEntropisalCalculator(crafted_words)
    df = calc.compute_all(["ab"])

    # Boundary-padded "#ab#" has target positions 1, 2, 3 -> 3 rows.
    assert len(df) == 3
    assert list(df.columns) == [
        "token_index",
        "word",
        "position",
        "target",
        "surprisal_1",
        "surprisal_2",
        "surprisal_3",
        "entropy_reduction_1",
        "entropy_reduction_2",
        "entropy_reduction_3",
        "entropy_difference_1",
        "entropy_difference_2",
        "entropy_difference_3",
        "surprisal_1_available",
        "surprisal_2_available",
        "surprisal_3_available",
        "entropy_reduction_1_available",
        "entropy_reduction_2_available",
        "entropy_reduction_3_available",
        "entropy_difference_1_available",
        "entropy_difference_2_available",
        "entropy_difference_3_available",
    ]


def test_compute_all_empty_input(sample_words):
    """compute_all on empty token list returns empty DataFrame with the right columns."""
    calc = CharacterEntropisalCalculator(sample_words)
    df = calc.compute_all([])
    assert len(df) == 0
    assert "entropy_reduction_2" in df.columns
    assert "surprisal_3" in df.columns


def test_entropy_reduction_positive(crafted_words):
    """H(A) - H(B) is positive when observing c_{i-1} sharpens the distribution (n=2)."""
    calc = CharacterEntropisalCalculator(crafted_words)
    df = calc.entropy_reduction(["ab"], n=2)
    row = df[df["position"] == 2].iloc[0]

    # A = H(target_at_2 | '#') marginalizing word[1] = {b:1, c:2, a:1} -> 1.5 bits
    # B = H(target_at_2 | '#a') = {b:1, c:1} -> 1.0 bit
    expected = _entropy([1, 2, 1]) - _entropy([1, 1])
    assert row["entropy_reduction"] == pytest.approx(expected)
    assert expected > 0
    assert bool(row["available"]) is True


def test_entropy_reduction_n3(sample_words):
    """n=3 (trigraph) reduction shape: returns one row per target char position."""
    calc = CharacterEntropisalCalculator(sample_words)
    df = calc.entropy_reduction(["the"], n=3)
    # "#the#" has target positions 1, 2, 3, 4 -> 4 rows
    assert len(df) == 4
    # Positions 1 and 2 lack enough context for trigraph -> unavailable
    row1 = df[df["position"] == 1].iloc[0]
    row2 = df[df["position"] == 2].iloc[0]
    assert bool(row1["available"]) is False
    assert bool(row2["available"]) is False
    assert math.isnan(row1["entropy_reduction"])


def test_entropy_reduction_invalid_n(sample_words):
    """Context lengths outside 1..3 are rejected for entropy reduction."""
    calc = CharacterEntropisalCalculator(sample_words)
    with pytest.raises(ValueError):
        calc.entropy_reduction(["the"], n=0)
    with pytest.raises(ValueError):
        calc.entropy_reduction(["the"], n=4)


def test_entropy_reduction_n1_uses_marginal(crafted_words):
    """n=1 entropy reduction = H(c) - H(c | c_{i-1}); a marginal-baseline MI.

    crafted_words = {"ab", "ac", "ba", "bc"}, each count 1. Boundary-padded forms
    "#ab#", "#ac#", "#ba#", "#bc#". For target at position 2 of "#ab#" (target='b'),
    n=1 conditions on c_{i-1} = word[1] = 'a' (single-char context, not bigraph):
      - transitions['a'] = {b:1, c:1, #:1}  (from "#ab#" i=1, "#ac#" i=1, "#ba#" i=2)
      - H(c | 'a') = H({1, 1, 1}) = log2(3) ~ 1.585 bits.
      - ER_1 = char_marginal_entropy - log2(3).
    """
    calc = CharacterEntropisalCalculator(crafted_words)
    df = calc.entropy_reduction(["ab"], n=1, signed=True)
    row = df[df["position"] == 2].iloc[0]
    expected = calc.char_marginal_entropy - math.log2(3)
    assert row["entropy_reduction"] == pytest.approx(expected)
    assert bool(row["available"]) is True


def test_entropy_reduction_n1_unavailable_at_position_one(sample_words):
    """At position 1 (first char after boundary), n=1 needs c_{i-1}='#'.

    Position 1's preceding char is the boundary '#'. H(target | '#') IS attested in
    the lookup, so n=1 ER should be available at position 1 (the boundary is a real
    context character). We check that the value is finite when attested.
    """
    calc = CharacterEntropisalCalculator(sample_words)
    df = calc.entropy_reduction(["cat"], n=1)
    row = df[df["position"] == 1].iloc[0]
    assert bool(row["available"]) is True
    assert not math.isnan(row["entropy_reduction"])


def test_entropy_reduction_deterministic_is_zero(deterministic_words):
    """When the dropped char is deterministic given the gap context, ER == 0 (n=2)."""
    calc = CharacterEntropisalCalculator(deterministic_words)
    # All words start with 'd' after '#'. So word[1] is deterministic given word[0]='#'.
    df = calc.entropy_reduction(["de"], n=2)
    row = df[df["position"] == 2].iloc[0]
    assert row["entropy_reduction"] == pytest.approx(0.0, abs=1e-12)
    assert bool(row["available"]) is True


def test_entropy_reduction_signed_can_be_negative(peaked_words):
    """Signed reduction is negative when the new context broadens expectations; clipped -> 0."""
    calc = CharacterEntropisalCalculator(peaked_words)
    # Target at position 2 of "#pa#": gap '#' is peaked (mostly 'a' via 'qa');
    # bigraph '#p' is uniform over {a, b} -> H = 1.0. So ER = small - 1.0 < 0.
    signed = calc.entropy_reduction(["pa"], n=2, signed=True)
    signed_row = signed[signed["position"] == 2].iloc[0]
    assert signed_row["entropy_reduction"] < 0
    assert bool(signed_row["available"]) is True

    clipped = calc.entropy_reduction(["pa"], n=2)  # default clipped
    clipped_row = clipped[clipped["position"] == 2].iloc[0]
    assert clipped_row["entropy_reduction"] == pytest.approx(0.0)


def test_entropy_reduction_unattested_is_na(sample_words):
    """Unattested gap or full contexts yield NaN and availability flag False."""
    calc = CharacterEntropisalCalculator(sample_words)
    df = calc.entropy_reduction(["xyz"], n=2)
    # word[1]='y', word[2]='z'. Bigraph '#x' is unattested in sample_words.
    row = df[df["position"] == 2].iloc[0]
    assert math.isnan(row["entropy_reduction"])
    assert bool(row["available"]) is False


def test_entropy_difference_shift_within_word(crafted_words):
    """entropy_difference shifts ent_n within a word; first scorable position is NaN."""
    calc = CharacterEntropisalCalculator(crafted_words)
    df = calc.entropy_difference(["ab", "ba"], n=1, signed=True)

    # Word "ab" -> "#ab#": position 1 has no predecessor -> NaN.
    first_of_word0 = df[(df["token_index"] == 0) & (df["position"] == 1)].iloc[0]
    assert math.isnan(first_of_word0["entropy_difference"])
    assert bool(first_of_word0["available"]) is False

    # Word "ba" -> "#ba#": position 1 also no predecessor (boundary not crossed from word 0).
    first_of_word1 = df[(df["token_index"] == 1) & (df["position"] == 1)].iloc[0]
    assert math.isnan(first_of_word1["entropy_difference"])
    assert bool(first_of_word1["available"]) is False


def test_entropy_difference_invalid_n(sample_words):
    """Context lengths outside 1..3 are rejected for entropy difference."""
    calc = CharacterEntropisalCalculator(sample_words)
    with pytest.raises(ValueError):
        calc.entropy_difference(["the"], n=0)
    with pytest.raises(ValueError):
        calc.entropy_difference(["the"], n=4)


def test_base_parameter_scales_values(crafted_words):
    """Changing the log base rescales information units by log(2)/log(base)."""
    calc = CharacterEntropisalCalculator(crafted_words)
    base2 = calc.entropy_reduction(["ab"], n=2, base=2.0)
    base4 = calc.entropy_reduction(["ab"], n=2, base=4.0)

    v2 = base2[base2["position"] == 2].iloc[0]["entropy_reduction"]
    v4 = base4[base4["position"] == 2].iloc[0]["entropy_reduction"]
    # 0.5 bit -> 0.25 in base-4 units
    assert v2 == pytest.approx(0.5)
    assert v4 == pytest.approx(0.25)


def test_calculate_metrics_includes_new_keys(crafted_words):
    """Aggregate metrics include the new entropy_reduction / entropy_difference keys."""
    calc = CharacterEntropisalCalculator(crafted_words)
    metrics = calc.calculate_metrics(["ab", "ac"])

    assert "char_entropy_reduction_2_support" in metrics
    assert "char_entropy_reduction_3_support" in metrics
    assert "char_entropy_difference_1_support" in metrics
    assert "char_entropy_difference_3_support" in metrics
    # At least one attested position for n=2 (target at pos 2 with '#a' bigraph).
    assert metrics["char_entropy_reduction_2_support"] >= 1
    assert "char_entropy_reduction_2" in metrics


def test_surprisal_default_uses_trigraph_context(sample_words):
    """Per-position surprisal defaults to the full trigraph context (n=3)."""
    calc = CharacterEntropisalCalculator(sample_words)
    df = calc.surprisal(["cat"])
    # "#cat#": positions 1 and 2 lack 3 preceding chars -> NaN.
    row1 = df[df["position"] == 1].iloc[0]
    row2 = df[df["position"] == 2].iloc[0]
    assert math.isnan(row1["surprisal"])
    assert math.isnan(row2["surprisal"])
    assert bool(row1["surprisal_available"]) is False
    # Position 3 ('t' given trigraph '#ca') should be attested.
    row3 = df[df["position"] == 3].iloc[0]
    assert bool(row3["surprisal_available"]) is True
    assert row3["surprisal"] is not None


def test_surprisal_n_parameter(sample_words):
    """surprisal(n=) selects the matching conditioning context length."""
    calc = CharacterEntropisalCalculator(sample_words)

    def avail_at(df, position):
        return bool(df[df["position"] == position].iloc[0]["surprisal_available"])

    df1 = calc.surprisal(["cat"], n=1)
    df2 = calc.surprisal(["cat"], n=2)
    df3 = calc.surprisal(["cat"], n=3)
    # n=1: positions 1, 2, 3 all have >=1 preceding char and should be attested.
    assert avail_at(df1, 1) and avail_at(df1, 2) and avail_at(df1, 3)
    # n=2: position 1 lacks 2 preceding chars -> unavailable.
    assert not avail_at(df2, 1)
    assert avail_at(df2, 2) and avail_at(df2, 3)
    # n=3: positions 1 and 2 lack 3 preceding chars -> unavailable.
    assert not avail_at(df3, 1) and not avail_at(df3, 2)
    assert avail_at(df3, 3)


def test_surprisal_invalid_n(sample_words):
    """Context lengths outside 1..3 are rejected for surprisal."""
    calc = CharacterEntropisalCalculator(sample_words)
    with pytest.raises(ValueError):
        calc.surprisal(["cat"], n=0)
    with pytest.raises(ValueError):
        calc.surprisal(["cat"], n=4)
