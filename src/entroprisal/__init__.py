"""entroprisal - Calculate entropy and surprisal linguistic metrics on text.

This package provides tools for calculating various entropy and surprisal metrics
on text based on reference corpora, including:
- Token-level n-gram entropy and surprisal
- Character-level transition entropy and surprisal
- Rest-of-word entropy and surprisal (bidirectional character-level)

Example usage:
    >>> from entroprisal import TokenEntropisalCalculator, CharacterEntropisalCalculator
    >>> from entroprisal.utils import load_4grams, load_google_books_words

    >>> # Token-level entropy and surprisal
    >>> ngrams = load_4grams("aw")
    >>> token_calc = TokenEntropisalCalculator(ngrams, min_frequency=100)
    >>> metrics = token_calc.calculate_metrics(["the", "cat", "sat"])

    >>> # Character-level entropy and surprisal
    >>> words_df = load_google_books_words()
    >>> char_calc = CharacterEntropisalCalculator(words_df)
    >>> metrics = char_calc.calculate_metrics("The cat sat on the mat")
"""

__version__ = "0.1.0"

from .character_entroprisal import CharacterEntropisalCalculator
from .rest_of_word_entroprisal import RestOfWordEntropisalCalculator
from .token_entroprisal import TokenEntropisalCalculator
from .utils import entropy

__all__ = [
    "TokenEntropisalCalculator",
    "CharacterEntropisalCalculator",
    "RestOfWordEntropisalCalculator",
    "entropy",
    "__version__",
]
