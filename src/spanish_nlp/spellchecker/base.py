from abc import ABC, abstractmethod
import re
from typing import List, Tuple, Set

class SpellCheckerBase(ABC):
    """
    Abstract Base Class for spell checker implementations.

    Defines the common interface that all spell checkers must adhere to.
    """

    def __init__(self, **kwargs):
        """
        Base initializer. Can be used for common setup.
        Accepts arbitrary keyword arguments for subclass flexibility.
        """
        pass

    @abstractmethod
    def is_correct(self, word: str) -> bool:
        """
        Checks if a word is considered correctly spelled by this checker.

        Args:
            word (str): The word to check.

        Returns:
            bool: True if the word is deemed correct, False otherwise.
        """
        pass

    @abstractmethod
    def suggest(self, word: str) -> List[str]:
        """
        Suggests corrections for a potentially misspelled word.

        Args:
            word (str): The word for which to get suggestions.

        Returns:
            List[str]: A list of correction suggestions.
        """
        pass

    @abstractmethod
    def correct_word(self, word: str) -> str:
        """
        Returns the most likely correction for a single word.

        Args:
            word (str): The word to correct.

        Returns:
            str: The corrected word, or the original word if no correction is
                 found or deemed necessary.
        """
        pass

    def _simple_tokenizer(self, text: str) -> List[Tuple[str, bool]]:
        """
        Simple tokenizer distinguishing words from non-words.

        Args:
            text (str): The input text.

        Returns:
            List[Tuple[str, bool]]: A list of tuples, where each tuple contains
                                     the token (str) and a boolean indicating
                                     if it's a word (True) or not (False).
        """
        tokens = []
        for match in re.finditer(r'(\b\w+\b)|(\W+)', text):
            word_match = match.group(1)
            non_word_match = match.group(2)
            if word_match:
                tokens.append((word_match, True))
            elif non_word_match:
                 tokens.append((non_word_match, False))
        return tokens

    def find_errors(self, text: str) -> List[str]:
        """
        Finds all unique potentially misspelled words in a given text.

        Args:
            text (str): The text to check for errors.

        Returns:
            List[str]: A list of unique words identified as potential errors.
        """
        errors: Set[str] = set()
        tokens = self._simple_tokenizer(text)
        for token, is_word in tokens:
            if is_word and not self.is_correct(token):
                errors.add(token)
        return list(errors)

    def correct_text(self, text: str) -> str:
        """
        Attempts to automatically correct an entire text. Use with caution,
        as automatic correction can introduce errors.

        Args:
            text (str): The text to correct.

        Returns:
            str: The text with corrections applied based on correct_word().
        """
        corrected_parts: List[str] = []
        tokens = self._simple_tokenizer(text)
        for token, is_word in tokens:
            if is_word:
                original_word = token
                corrected_word = self.correct_word(original_word)

                if corrected_word:
                    if original_word.istitle():
                        corrected_word = corrected_word.title()
                    elif original_word.isupper():
                        if len(original_word) > 1 or len(corrected_word) == 1:
                             corrected_word = corrected_word.upper()
                        elif len(original_word) == 1 and len(corrected_word) > 1:
                             corrected_word = corrected_word
                else:
                    corrected_word = original_word

                corrected_parts.append(corrected_word)
            else:
                corrected_parts.append(token)

        return "".join(corrected_parts)
