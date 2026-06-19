# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.7.0] - 2026-06-19

### Changed
- **Default `min_frequency` lowered from 100 to 5** for `TokenEntropisalCalculator`. The previous default of 100 discarded the long tail of attested n-grams, leaving many tokens' contexts unscored (low `*_support`, more `NaN`); `min_frequency=1` keeps the full table but, for the all-words 4-grams (~237M rows), needs tens of GB of RAM to build. The new default of `5` retains far more coverage than `100` (≈3.7M vs ≈47K all-word 4-grams) while remaining buildable on a laptop. **Metric values and support counts change for anyone who relied on the previous default**; pass `min_frequency=100` explicitly to reproduce pre-0.7 results. The threshold still controls only which n-grams are available — no smoothing or backoff is applied.
- **`TokenEntropisalCalculator.calculate_batch` is now vectorized.** Instead of re-joining each document against the lookup tables, it assembles the whole corpus into a single per-position table and joins each lookup once. Results are identical to the previous per-document path up to floating-point summation order (max abs difference ≈ 1e-14, with identical `NaN` patterns and `*_support` counts). On multi-thousand-document corpora this is dramatically faster — the per-position metric computation over the CLEAR corpus (4,724 excerpts) dropped from hours to a few seconds.
- **Build-time intermediates are released after construction.** Once the lookup tables are built, `__init__` clears `_token_transitions` and sets `ngram_df` to `None` (neither is read by any query method), lowering the calculator's resident memory footprint. Note: `calc.ngram_df` is no longer available after construction — code that inspected it should filter the source frame directly instead. This lowers steady-state memory but does not change the peak reached while the lookups are being built.

## [0.6.0] - 2026-06-05

### Removed
- **`entropy_difference_1` and `entropy_reduction_1`** (both token- and character-level) are no longer computed. `ENTROPY_DIFFERENCE_NS` and `ENTROPY_REDUCTION_NS` are now `(2, 3)`; the `*_1` keys, columns, and `_support` counts are dropped from `calculate_metrics` and `compute_all`, and `entropy_difference(..., n=1)` / `entropy_reduction(..., n=1)` now raise `ValueError`. Both n=1 cases are degenerate at the n=1 conditioning length:
  - `entropy_difference_1`: within a word/sequence the position-to-position differences telescope to `E_1[first] − E_1[last]`; since the initial term (`H(unit | boundary)`, a corpus constant) is identical for every unit, the measure reduces to a negated, length-normalized function of the *final* unit's branching entropy. `char_entropy_difference_1` correlated r ≈ −1.00 with the word-final character's entropy (signed/unclipped) and carried a strong spurious word-length dependence.
  - `entropy_reduction_1`: at n=1 the Distribution A baseline is the scalar marginal entropy, so the per-position reduction `H(unit) − H(unit | preceding)` averages to an exact affine function of the mean transitional entropy (`char_entropy_reduction_1 = marginal − char_entropy`, r = −1.00) — a sign-flipped, shifted copy of an existing metric rather than a distinct construct.

  `entropy_difference_2/3` and `entropy_reduction_2/3` condition on longer contexts, do not collapse to a single-unit lookup, and are unaffected. The directional `{lr,rl}_entropy_reduction_1` metrics of `RestOfWordEntropisalCalculator` are intentionally left unchanged.

## [0.5.1] - 2026-06-01

### Fixed
- `RestOfWordEntropisalCalculator` no longer emits degenerate `c2`/`c3` entropy and surprisal values for words too short to support the conditioning context. Because words are boundary-padded (`#word#`), the `(n+1)`-character context window for a token with fewer than `n` content characters overran into the opposite boundary, collapsing the predicted "rest" to `""` and producing a constant entropy 0 / surprisal 0. Concretely, every 1-character word (e.g. `a`, `i`) produced a spurious zero for `lr/rl_c2_*`, and every 2-character word (e.g. `to`, `of`, `is`) produced one for `lr/rl_c3_*`. `c_n` is now computed only when the token has at least `n` content characters (`len(padded_word) >= n + 2`); shorter words contribute nothing to the aggregate (absent from `*_support`) and yield `NaN` in the per-position `surprisal`/`entropy_reduction`/`compute_all` outputs (which also removes the degenerate term from per-position `entropy_reduction_2`/`entropy_reduction_3` at those positions). Note: aggregate `lr/rl_c2_entropy`, `lr/rl_c3_entropy`, and their surprisal counterparts may change for inputs containing 1- or 2-character words, since the spurious zeros are no longer averaged in.

## [0.5.0] - 2026-05-29

### Fixed
- `preprocess_text` now keeps only alphabetic tokens (`token.is_alpha`) in the default (`content_words_only=False`) path, matching how the reference n-gram corpus was tokenized. Previously punctuation, whitespace, digits, and spaCy-split clitics (`n't`, `'s`) were retained — none of which exist in the reference vocabulary, so they inflated `n_tokens` and broke the n-gram contexts of adjacent real words. This aligns `n_tokens` with a true word count and improves metric coverage. Note: per-document metric values and `n_tokens` may change for inputs containing non-alphabetic tokens.

## [0.3.1] - 2026-05-20

### Changed
- README updates only.

## [0.3.0] - 2026-05-20

### Added
- `TokenEntropisalCalculator.entropy_reduction(tokens, *, n=3, signed=False, base=2.0)`: per-position Hale-style conditional mutual information `H(W_t | w_{t-n}..w_{t-2}) - H(W_t | w_{t-n}..w_{t-1})` for `n` in {2, 3}.
- `TokenEntropisalCalculator.entropy_difference(tokens, *, n=3, signed=False, base=2.0)`: per-position Lowder-style entropy difference `E_n[t-1] - E_n[t]` for `n` in {1, 2, 3}.
- `TokenEntropisalCalculator.compute_all(tokens, ...)`: convenience method returning all per-position metrics at every context length in one DataFrame.
- New aggregate keys in `TokenEntropisalCalculator.calculate_metrics`: `entropy_reduction_{2,3}`, `entropy_difference_{1,2,3}` (plus matching `_support` counts).

## [0.4.0] - 2026-05-26

### Added
- **Per-position metrics for `CharacterEntropisalCalculator`**: `surprisal`, `entropy_reduction`, `entropy_difference`, and `compute_all` now return per-character-position DataFrames, mirroring the token-level API.
- **Per-word metrics for `RestOfWordEntropisalCalculator`**: `surprisal`, `entropy_reduction`, and `compute_all` return per-word DataFrames, parameterized by `direction` (`"lr"` or `"rl"`).
- **Character-level entropy reduction** (Hale-style conditional mutual information): `H(c_i | c_{i-n}..c_{i-2}) - H(c_i | c_{i-n}..c_{i-1})` for `n` in {1, 2, 3}.
- **Character-level entropy difference** (Lowder-style): `E_n[i-1] - E_n[i]` within each word, for `n` in {1, 2, 3}.
- **Rest-of-word entropy reduction** in both directions: `H(W | (n-1) chars) - H(W | n chars)` for `n` in {1, 2, 3}. Because the conditioning target is the same (word identity) across prefix lengths, this is the clean Hale-style CMI directly parallel to Hale (2016)'s parse-given-words formulation.
- **`n=1` entropy reduction** for `TokenEntropisalCalculator` and `CharacterEntropisalCalculator`, using the marginal target entropy as the Distribution A baseline. Reduces to mutual information between adjacent tokens / characters.
- **`n` parameter on per-position `surprisal()`** for token and character calculators (`n` in {1, 2, 3}; default `n=3`).
- **Marginal-entropy attributes**: `TokenEntropisalCalculator.token_marginal_entropy`, `CharacterEntropisalCalculator.char_marginal_entropy`, `RestOfWordEntropisalCalculator.word_marginal_entropy`.
- New aggregate keys in `calculate_metrics`: `char_entropy_reduction_{1,2,3}`, `char_entropy_difference_{1,2,3}`, `{lr,rl}_entropy_reduction_{1,2,3}` (plus matching `_support` counts).
- Metrics-at-a-glance table in the README.

### Changed
- **BREAKING**: `TokenEntropisalCalculator.compute_all` and `CharacterEntropisalCalculator.compute_all` now expose `surprisal_1`, `surprisal_2`, `surprisal_3` columns instead of a single `surprisal` column. The standalone `surprisal()` method still returns a column named `surprisal`.
- `TokenEntropisalCalculator.ENTROPY_REDUCTION_NS` is now `(1, 2, 3)` (was `(2, 3)`).
- `CharacterEntropisalCalculator.ENTROPY_REDUCTION_NS` is now `(1, 2, 3)` (was previously undefined; entropy reduction is new at this level).

### Fixed
- Notebook examples (`examples/usage_examples.ipynb`) previously passed raw strings to `CharacterEntropisalCalculator.calculate_metrics` and `RestOfWordEntropisalCalculator.calculate_metrics` (and their batch variants). Strings silently iterated as character "tokens", producing incorrect results. Cells now tokenize via `preprocess_text` first.

### Migration
The only breaking change is the `compute_all` column rename on token and character calculators:
```python
# Before:
df = calc.compute_all(tokens)
df["surprisal"]       # was the 4-gram (or trigraph) surprisal

# After:
df = calc.compute_all(tokens)
df["surprisal_3"]     # 4-gram (or trigraph) surprisal — same values as before
df["surprisal_2"]     # trigram (or bigraph) surprisal — newly available
df["surprisal_1"]     # bigram (or single-char) surprisal — newly available
```
The standalone `calc.surprisal(tokens)` method is unchanged at its default (`n=3`); use `calc.surprisal(tokens, n=1)` or `n=2` to access the shorter contexts.

## [0.2.3] - 2025-12-10

### Changed
- **BREAKING**: All calculators now require `List[str]` (token list) as input instead of `str`
- **BREAKING**: `calculate_batch()` now requires `List[List[str]]` (list of token lists)
- This aligns all calculators with `preprocess_text()` output format
- Eliminates ambiguity between batch of strings vs. list of tokens

### Migration
Before:
```python
calc.calculate_metrics("The quick brown fox")
calc.calculate_batch(["Text one", "Text two"])
```

After:
```python
from entroprisal import preprocess_text
tokens = preprocess_text("The quick brown fox")[0]
calc.calculate_metrics(tokens)

token_lists = preprocess_text(["Text one", "Text two"])
calc.calculate_batch(token_lists)
```

## [0.2.2] - 2025-12-10

### Fixed
- Fixed package build configuration so bundled data files are actually included in wheel
- Updated `.gitignore` to not exclude `src/entroprisal/data/` directory
- Added `__init__.py` to data directory to ensure it's treated as a package

## [0.2.1] - 2025-12-10

### Fixed
- Google Books word frequency data is now properly bundled with the package
- No longer attempts to download bundled data files from Hugging Face Hub
- Uses `importlib.resources` for access to bundled data in installed packages

## [0.2.0] - 2025-12-09

### Added
- `preprocess_text()` function for spaCy-based tokenization with content word filtering
- `is_content_token()` helper function for identifying content words (nouns, verbs, adjectives, adverbs)
- Automatic data file downloads from Hugging Face Hub with fallback to direct HTTP downloads
- `entropy()` function exported at package level for custom calculations
- Surprisal metrics to `CharacterEntropisalCalculator` and `RestOfWordEntropisalCalculator`
- Lookup methods for surprisal values: `get_character_surprisal()`, `get_bigraph_surprisal()`, `get_trigraph_surprisal()`
- Support count metrics for all entropy and surprisal calculations

### Changed
- Renamed package from internal names to "entroprisal" terminology throughout
- Removed `preprocess` parameter from `calculate_metrics()` and `calculate_batch()` methods
  - Users should now use `preprocess_text()` for consistent text preprocessing
- Removed scipy dependency; entropy calculation now uses custom implementation

### Fixed
- Fixed `get_character_surprisal()` signature to accept context and target parameters

### Removed
- Removed inline preprocessing from calculators (use `preprocess_text()` instead)

## [0.1.0] - 2025-12-01

### Added
- Initial release
- `TokenEntropisalCalculator` for token-level n-gram entropy
- `CharacterEntropisalCalculator` for character-level transition entropy
- `RestOfWordEntropisalCalculator` for bidirectional rest-of-word entropy
- Reference data loading utilities (`load_google_books_words`, `load_4grams`)
- Batch processing support for all calculators
