# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
