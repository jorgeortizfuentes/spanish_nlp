import logging
from spellchecker import SpellChecker
from typing import List, Optional, Union

from .base import SpellCheckerBase

logger = logging.getLogger(__name__)

class DictionarySpellChecker(SpellCheckerBase):
    """
    Spell checker implementation using the 'pyspellchecker' library.
    Relies on dictionary lookups and edit distance.
    """

    def __init__(self,
                 language: str = 'es',
                 distance: int = 2,
                 custom_dictionary: Optional[Union[str, List[str]]] = None,
                 **kwargs):
        """
        Initializes the dictionary-based spell checker.

        Args:
            language (str): Language code (e.g., 'es'). Defaults to 'es'.
            distance (int): Maximum edit distance for suggestions. Defaults to 2.
            custom_dictionary (Optional[Union[str, List[str]]]): Path to a text file
                or a list of words to add to the dictionary. Defaults to None.
            **kwargs: Additional arguments passed to the base class.
        """
        super().__init__(**kwargs)
        self.spell: Optional[SpellChecker] = None
        try:
            self.spell = SpellChecker(language=language, distance=distance)

            if custom_dictionary:
                if isinstance(custom_dictionary, str):
                    logger.info(f"Loading custom dictionary from file: {custom_dictionary}")
                    self.spell.word_frequency.load_text_file(custom_dictionary)
                elif isinstance(custom_dictionary, list):
                    logger.info(f"Loading {len(custom_dictionary)} words from custom list.")
                    self.spell.word_frequency.load_words(custom_dictionary)

            self._language = language
            logger.info(f"DictionarySpellChecker initialized for language '{language}' with distance {distance}.")

        except ValueError as e:
            logger.error(f"Failed to initialize SpellChecker for language '{language}': {e}", exc_info=True)
            logger.error("Please ensure the dictionary for this language is available for pyspellchecker.")
        except Exception as e:
            logger.error(f"An unexpected error occurred during DictionarySpellChecker initialization: {e}", exc_info=True)


    def is_correct(self, word: str) -> bool:
        """Checks if the word exists in the dictionary (case-insensitive)."""
        if not self.spell:
            logger.warning("DictionarySpellChecker not initialized properly, assuming word is correct.")
            return True
        return word.lower() in self.spell

    def suggest(self, word: str) -> List[str]:
        """Suggests corrections based on edit distance."""
        if not self.spell:
            logger.warning("DictionarySpellChecker not initialized properly, returning no suggestions.")
            return []
        try:
            candidates_set = self.spell.candidates(word.lower())
            # Explicitly check if the result is None before converting to list
            if candidates_set is None:
                 logger.warning(f"pyspellchecker returned None for candidates of '{word}'. Returning empty list.")
                 return []
            return list(candidates_set)
        except Exception as e:
             logger.error(f"Error calling pyspellchecker.candidates for '{word}': {e}", exc_info=True)
             return [] # Return empty list on any error during candidates call

    def correct_word(self, word: str) -> str:
        """Returns the most likely correction based on frequency and edit distance."""
        if not self.spell:
            logger.warning("DictionarySpellChecker not initialized properly, returning original word.")
            return word
        corrected = self.spell.correction(word.lower())
        return corrected if corrected is not None else word.lower()
