# entroprisal

Calculate information theoretic linguistic metrics on text using reference corpora.

## Overview

`entroprisal` is a Python package that computes various entropy and surprisal metrics for text analysis. It provides three main calculators:

- **TokenEntropisalCalculator**: Token-level n-gram entropy and surprisal
- **CharacterEntropisalCalculator**: Character-level entropy and surprisal
- **RestOfWordEntropisalCalculator**: Character-level rest-of-word entropy and surprisal (bidirectional: left-to-right and right-to-left word completion)

These metrics are useful for analyzing text complexity, readability, and information content.

## Installation

### Basic Installation

```bash
pip install entroprisal[all]
```

The package will automatically download reference data files from Hugging Face Hub when first used (~4GiB total).

SpaCy and Hugging Face Hub are optional dependencies for additional functionality. A minimal installation without these dependencies is also possible:

```bash
pip install entroprisal
```

### Optional Dependencies included in `all`

`huggingface-hub` is used for faster downloads with caching (recommended)

`spacy` is used for classifying content words vs. function words in your target text

If using SpaCy, you will need to download a SpaCy language model as well:

```bash
python -m spacy download en_core_web_lg
```

### Development Installation

```bash
# Clone the repository
git clone https://github.com/learlab/entroprisal.git
cd entroprisal

# Install in editable mode with dev dependencies
uv pip install -e .[dev]
```

## Data Files

Reference corpus files are automatically downloaded from [Hugging Face Hub](https://huggingface.co/datasets/learlab/entroprisal-data) on first use:

- `google-books-dictionary-words.txt` - Word frequencies (included in package)
- `4grams_aw.parquet` - All-word 4-gram frequencies (~2GiB)
- `4grams_cw.parquet` - Content-word 4-gram frequencies (~1.8GiB)

Files are cached locally to avoid re-downloading. To use the faster Hugging Face Hub downloader with resume capability, install with `pip install entroprisal[hf]`.

## Quick Start

### Token-Level Entropy and Surprisal

```python
from entroprisal import TokenEntropisalCalculator
from entroprisal.utils import load_4grams

# Load reference n-gram data
ngrams = load_4grams("aw")  # "aw" = all words, "cw" = content words

# Initialize calculator
calc = TokenEntropisalCalculator(ngrams, min_frequency=100)

# Calculate metrics for a list of tokens
tokens = ["the", "quick", "brown", "fox"]
metrics = calc.calculate_metrics(tokens)

print(metrics)
# Output includes:
# - ngram_surprisal_1, ngram_surprisal_2, ngram_surprisal_3
# - ngram_entropy_1, ngram_entropy_2, ngram_entropy_3
# - Support counts for each metric
```

### Character-Level Entropy and Surprisal

```python
from entroprisal import CharacterEntropisalCalculator
from entroprisal.utils import load_google_books_words

# Load word frequency data
words_df = load_google_books_words()

# Initialize calculator
calc = CharacterEntropisalCalculator(words_df)

# Calculate metrics for text
text = "The quick brown fox jumps over the lazy dog"
metrics = calc.calculate_metrics(text)

print(metrics)
# Output includes:
# - char_entropy, char_surprisal: Single character transition metrics
# - bigraph_entropy, bigraph_surprisal: Two-character context metrics
# - trigraph_entropy, trigraph_surprisal: Three-character context metrics
```

### Rest-of-Word Entropy and Surprisal (Character-Level, Bidirectional)

```python
from entroprisal import RestOfWordEntropisalCalculator
from entroprisal.utils import load_google_books_words

# Load word frequency data
words_df = load_google_books_words()

# Initialize calculator
calc = RestOfWordEntropisalCalculator(words_df)

# Calculate metrics for text
text = "The quick brown fox"
metrics = calc.calculate_metrics(text)

print(metrics)
# Output includes:
# - lr_c1_entropy, lr_c1_surprisal: Left-to-right, 1-char context
# - lr_c2_entropy, lr_c2_surprisal: Left-to-right, 2-char context
# - lr_c3_entropy, lr_c3_surprisal: Left-to-right, 3-char context
# - rl_c1_entropy, rl_c1_surprisal: Right-to-left, 1-char context
# - rl_c2_entropy, rl_c2_surprisal: Right-to-left, 2-char context
# - rl_c3_entropy, rl_c3_surprisal: Right-to-left, 3-char context
# - mean_word_length
```

### Batch Processing

All calculators support batch processing:

```python
# Process multiple texts at once
texts = [
    "First text sample",
    "Second text sample",
    "Third text sample"
]

# Returns a pandas DataFrame with one row per text
results_df = calc.calculate_batch(texts)
print(results_df)
```

## API Reference

### TokenEntropisalCalculator

Calculate token-level entropy and surprisal metrics using n-gram frequencies.

**Methods:**

- `calculate_metrics(tokens: List[str]) -> Dict[str, float]`: Calculate metrics for a token list
- `calculate_batch(token_lists: List[List[str]]) -> pd.DataFrame`: Batch processing
- `get_detailed_ngram_analysis(tokens: List[str]) -> Dict[int, pd.DataFrame]`: Detailed per-token analysis

### CharacterEntropisalCalculator

Calculate character-level transition entropy and surprisal.

**Methods:**

- `calculate_metrics(text: str, preprocess: bool = True) -> Dict[str, float]`: Calculate metrics for text
- `calculate_batch(texts: List[str], preprocess: bool = True) -> pd.DataFrame`: Batch processing
- `get_character_entropy(char: str) -> Optional[float]`: Lookup entropy for specific character
- `get_character_surprisal(char: str) -> Optional[float]`: Lookup surprisal for specific character
- `get_bigraph_entropy(bigraph: str) -> Optional[float]`: Lookup entropy for bigraph
- `get_bigraph_surprisal(bigraph: str) -> Optional[float]`: Lookup surprisal for bigraph
- `get_trigraph_entropy(trigraph: str) -> Optional[float]`: Lookup entropy for trigraph
- `get_trigraph_surprisal(trigraph: str) -> Optional[float]`: Lookup surprisal for trigraph

### RestOfWordEntropisalCalculator

Calculate character-level rest-of-word entropy and surprisal in both directions (predicting remaining characters from left-to-right and right-to-left contexts).

**Methods:**

- `calculate_metrics(text: str, preprocess: bool = True) -> Dict[str, float]`: Calculate metrics for text
- `calculate_batch(texts: List[str], preprocess: bool = True) -> pd.DataFrame`: Batch processing
- `get_word_frequency(word: str) -> int`: Get frequency of a word in reference corpus

## Utilities

```python
from entroprisal.utils import (
    load_google_books_words,
    load_4grams,
    get_data_dir,
    preprocess_text
)

# Load reference data
words_df = load_google_books_words()
ngrams_aw = load_4grams("aw")
ngrams_cw = load_4grams("cw")

# Get data directory path
data_dir = get_data_dir()

# Preprocess text
cleaned = preprocess_text("Text with punctuation!", aggressive=True)
```

## Examples

See `examples/usage_examples.ipynb` for comprehensive examples including:

- Loading and initializing calculators
- Processing single texts and batches
- Combining multiple metrics
- Visualizing results

## Development

### Running Tests

```bash
pytest tests/
```

### Code Style

```bash
# Format code
black src/

# Lint code
ruff check src/
```

## License

It's MIT licensed. Do what you want with it.

## Citation

On the other hand, if you are an academic, please cite!

```bibtex
@software{entroprisal,
  title = {entroprisal: Entropy-based linguistic metrics},
  author = {Langdon Holmes},
  year = {2025},
  url = {https://github.com/learlab/entroprisal}
}
```

## Acknowledgments

Reference data sources:

- Google Books word frequencies: [gwordlist](https://github.com/hackerb9/gwordlist)
- N-gram token frequencies: Derived from the slimpajama test set [slimpajama](https://www.cerebras.ai/blog/slimpajama-a-627b-token-cleaned-and-deduplicated-version-of-redpajama)
