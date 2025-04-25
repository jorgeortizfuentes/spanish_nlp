import logging
import torch
import re
from transformers import pipeline, AutoTokenizer, AutoModelForMaskedLM
from typing import List, Optional, Union

from .base import SpellCheckerBase

logger = logging.getLogger(__name__)

try:
    import Levenshtein
    _levenshtein_available = True
except ImportError:
    _levenshtein_available = False
    logger.info("Levenshtein library not found. Contextual suggestions will not use edit distance filtering.")


class ContextualLMSpellChecker(SpellCheckerBase):
    """
    Spell checker implementation using a contextual language model (e.g., BERT, BETO)
    via the Hugging Face transformers library. Correction is context-dependent.
    """

    DEFAULT_MODEL = "dccuchile/bert-base-spanish-wwm-uncased"

    def __init__(self,
                 model_name: str = DEFAULT_MODEL,
                 device: Optional[Union[str, int]] = None,
                 top_k: int = 5,
                 suggestion_distance_threshold: int = 2,
                 **kwargs):
        """
        Initializes the contextual language model spell checker.

        Args:
            model_name (str): Name or path of the pre-trained masked language model.
            device (Optional[Union[str, int]]): Device ('cpu', 'cuda', cuda index). Auto-detects if None.
            top_k (int): Number of suggestions to consider from the model.
            suggestion_distance_threshold (int): Only suggest corrections within this edit distance.
                                                 Requires 'python-Levenshtein'.
            **kwargs: Additional arguments passed to the base class.
        """
        super().__init__(**kwargs)
        self.pipeline = None
        self.tokenizer = None
        self.mask_token: Optional[str] = None
        self.top_k = top_k

        if not _levenshtein_available:
             logger.warning("Levenshtein library not found. Edit distance filtering disabled.")
             self.suggestion_distance_threshold = None
        else:
             self.suggestion_distance_threshold = suggestion_distance_threshold


        if device is None:
            resolved_device = 0 if torch.cuda.is_available() else -1
            device_name = "cuda" if resolved_device == 0 else "cpu"
        elif isinstance(device, int):
             resolved_device = device
             device_name = f"cuda:{device}"
        else:
             resolved_device = 0 if device == 'cuda' else -1
             device_name = device

        logger.info(f"Initializing ContextualLMSpellChecker with model '{model_name}' on device '{device_name}'.")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForMaskedLM.from_pretrained(model_name)
            self.mask_token = self.tokenizer.mask_token

            self.pipeline = pipeline(
                "fill-mask",
                model=model,
                tokenizer=self.tokenizer,
                device=resolved_device,
                top_k=self.top_k
            )
            logger.info("ContextualLMSpellChecker initialized successfully.")

        except Exception as e:
            logger.error(f"Failed to initialize transformers pipeline for model '{model_name}': {e}", exc_info=True)
            self.pipeline = None
            self.tokenizer = None
            self.mask_token = None

    def is_correct(self, word: str) -> bool:
        """
        Checks if the word is known by the model's tokenizer vocabulary.
        Note: This is a weak check. Contextual checking happens in correct_text.
        """
        if not self.tokenizer:
            logger.warning("ContextualLMSpellChecker not initialized, assuming word is correct.")
            return True
        return word in self.tokenizer.vocab or word.lower() in self.tokenizer.vocab

    def suggest(self, word: str) -> List[str]:
        """
        Suggests corrections by masking the word and predicting replacements.
        WARNING: Lacks context. Use correct_text for context-aware corrections.
        """
        if not self.pipeline or not self.mask_token:
            logger.warning("ContextualLMSpellChecker not initialized, returning no suggestions.")
            return []

        try:
            results = self.pipeline(f"{self.mask_token}")
            suggestions = [r['token_str'].strip() for r in results if r['token_str'].strip()]

            if self.suggestion_distance_threshold is not None:
                filtered_suggestions = [
                    s for s in suggestions
                    if Levenshtein.distance(word.lower(), s.lower()) <= self.suggestion_distance_threshold
                ]
                return filtered_suggestions
            else:
                return suggestions

        except Exception as e:
            logger.error(f"Error during suggestion generation for '{word}': {e}", exc_info=True)
            return []

    def correct_word(self, word: str) -> str:
        """
        Returns the most likely replacement by masking the word.
        WARNING: Lacks context. Use correct_text for context-aware corrections.
        """
        suggestions = self.suggest(word)
        if suggestions and suggestions[0].lower() != word.lower():
            if re.match(r'^\w+$', suggestions[0]):
                 return suggestions[0]
        return word

    def correct_text(self, text: str) -> str:
        """
        Corrects text contextually by masking each word and predicting replacements.
        """
        if not self.pipeline or not self.tokenizer or not self.mask_token:
            logger.warning("ContextualLMSpellChecker not initialized, returning original text.")
            return text

        tokens_info = self._simple_tokenizer(text)
        original_words = [token for token, is_word in tokens_info if is_word]

        if not original_words:
            return text

        corrected_word_list = list(original_words) # Create a mutable list of words

        for i, word_to_check in enumerate(original_words):
            # Create context with mask for the current word
            temp_tokens = list(original_words)
            temp_tokens[i] = self.mask_token
            masked_context = " ".join(temp_tokens)

            try:
                results = self.pipeline(masked_context)
                if results:
                    top_suggestion = results[0]['token_str'].strip()

                    if not top_suggestion or not re.match(r'^\w+$', top_suggestion):
                        continue

                    is_different = word_to_check.lower() != top_suggestion.lower()
                    is_close_enough = True
                    if self.suggestion_distance_threshold is not None and is_different:
                        is_close_enough = Levenshtein.distance(word_to_check.lower(), top_suggestion.lower()) <= self.suggestion_distance_threshold

                    if is_different and is_close_enough:
                        # Update the word in our mutable list
                        corrected_word_list[i] = top_suggestion

            except Exception as e:
                 logger.warning(f"Error processing word '{word_to_check}' in context: {e}")
                 continue

        # Reconstruct the text using the corrected word list and original non-words
        final_parts = []
        word_idx = 0
        for token, is_word in tokens_info:
            if is_word:
                corrected_word = corrected_word_list[word_idx]
                # Preserve case
                if token.istitle():
                    corrected_word = corrected_word.title()
                elif token.isupper():
                     if len(token) > 1 or len(corrected_word) == 1:
                          corrected_word = corrected_word.upper()
                     elif len(token) == 1 and len(corrected_word) > 1:
                          corrected_word = corrected_word # Keep correction case
                     else: # Both long, original was upper
                          corrected_word = corrected_word.upper()

                final_parts.append(corrected_word)
                word_idx += 1
            else:
                final_parts.append(token)

        return "".join(final_parts)
