import logging
from typing import List, Optional, Union

from .base import SpellCheckerBase
from .dictionary_impl import DictionarySpellChecker
from .contextual_lm_impl import ContextualLMSpellChecker

logger = logging.getLogger(__name__)

_IMPLEMENTATIONS = {
    'dictionary': DictionarySpellChecker,
    'contextual_lm': ContextualLMSpellChecker,
}

class SpanishSpellChecker:
    """
    Main facade for Spanish spell checking.

    Provides access to different spell checking strategies (e.g., dictionary-based,
    contextual language model-based).
    """
    def __init__(self,
                 method: str = 'dictionary',
                 language: str = 'es',
                 **kwargs):
        """
        Initializes the spell checker using the specified method.

        Args:
            method (str): The spell checking method to use.
                          Available: 'dictionary', 'contextual_lm'.
                          Defaults to 'dictionary'.
            language (str): Language code, primarily used by 'dictionary' method.
                            Defaults to 'es'.
            **kwargs: Additional keyword arguments specific to the chosen method.
                      For 'dictionary': distance, custom_dictionary.
                      For 'contextual_lm': model_name, device, top_k, suggestion_distance_threshold.
        """
        implementation_class = _IMPLEMENTATIONS.get(method.lower())

        if not implementation_class:
            available = ", ".join(_IMPLEMENTATIONS.keys())
            raise ValueError(
                f"Spell checking method '{method}' not recognized. "
                f"Available methods: {available}"
            )

        logger.info(f"Initializing SpanishSpellChecker with method: '{method}'")

        if 'language' not in kwargs:
             kwargs['language'] = language

        try:
            self._impl: SpellCheckerBase = implementation_class(**kwargs)
        except Exception as e:
            logger.error(f"Failed to instantiate spell checker implementation '{method}': {e}", exc_info=True)
            raise RuntimeError(f"Could not initialize spell checker method '{method}'. See logs for details.") from e


    def is_correct(self, word: str) -> bool:
        """Checks if a word is correct using the selected method."""
        return self._impl.is_correct(word)

    def suggest(self, word: str) -> List[str]:
        """Suggests corrections for a word using the selected method."""
        return self._impl.suggest(word)

    def correct_word(self, word: str) -> str:
        """Gets the most likely correction for a word using the selected method."""
        return self._impl.correct_word(word)

    def find_errors(self, text: str) -> List[str]:
        """Finds potential errors in text using the selected method."""
        return self._impl.find_errors(text)

    def correct_text(self, text: str) -> str:
        """Corrects an entire text using the selected method."""
        return self._impl.correct_text(text)

    def get_implementation_details(self) -> str:
        """Returns information about the currently used implementation."""
        return f"Using implementation: {self._impl.__class__.__name__}"

__all__ = [
    "SpanishSpellChecker",
    "SpellCheckerBase",
    "DictionarySpellChecker",
    "ContextualLMSpellChecker"
]
