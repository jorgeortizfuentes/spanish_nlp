import logging
import torch
import re
from transformers import pipeline, AutoTokenizer, AutoModelForMaskedLM
from typing import List, Optional, Union

# Importar la implementación del diccionario
from .base import SpellCheckerBase
from .dictionary_impl import DictionarySpellChecker

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
    Combines LM suggestions with dictionary validation and Levenshtein distance.
    """

    DEFAULT_MODEL = "dccuchile/bert-base-spanish-wwm-uncased"

    def __init__(self,
                 model_name: str = DEFAULT_MODEL,
                 device: Optional[Union[str, int]] = None,
                 top_k: int = 10,
                 suggestion_distance_threshold: int = 2,
                 # Parámetros para el corrector de diccionario interno
                 dict_language: str = 'es',
                 dict_distance: int = 2,
                 dict_custom_dictionary: Optional[Union[str, List[str]]] = None,
                 **kwargs):
        """
        Initializes the contextual language model spell checker.

        Args:
            model_name (str): Name or path of the pre-trained masked language model.
            device (Optional[Union[str, int]]): Device ('cpu', 'cuda', cuda index). Auto-detects if None.
            top_k (int): Number of suggestions to consider from the model. Defaults to 10.
            suggestion_distance_threshold (int): Only suggest corrections within this edit distance.
                                                 Requires 'python-Levenshtein'. Defaults to 2.
            dict_language (str): Language for the internal dictionary checker. Defaults to 'es'.
            dict_distance (int): Max edit distance for the internal dictionary checker's suggestions (less critical here). Defaults to 2.
            dict_custom_dictionary (Optional[Union[str, List[str]]]): Custom dictionary for the internal checker.
            **kwargs: Additional arguments passed to the base class.
        """
        super().__init__(**kwargs)
        self.pipeline = None
        self.tokenizer = None
        self.mask_token: Optional[str] = None
        self.top_k = top_k
        self.model_name = model_name

        if not _levenshtein_available:
             logger.warning("Levenshtein library not found. Edit distance filtering disabled.")
             self.suggestion_distance_threshold = None
        else:
             self.suggestion_distance_threshold = max(0, suggestion_distance_threshold)


        if device is None:
            resolved_device = 0 if torch.cuda.is_available() else -1
            device_name = "cuda" if resolved_device == 0 else "cpu"
        elif isinstance(device, int):
             resolved_device = device
             device_name = f"cuda:{device}"
        else:
             if device.lower() == 'cuda' and torch.cuda.is_available():
                 resolved_device = 0
                 device_name = 'cuda'
             else:
                 resolved_device = -1
                 device_name = 'cpu'


        logger.info(f"Initializing ContextualLMSpellChecker with model '{model_name}' on device '{device_name}'.")
        logger.info(f"LM top_k={self.top_k}, Levenshtein threshold={self.suggestion_distance_threshold}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForMaskedLM.from_pretrained(model_name)
            if resolved_device != -1 and isinstance(resolved_device, int):
                 model.to(f'cuda:{resolved_device}')

            self.mask_token = self.tokenizer.mask_token

            self.pipeline = pipeline(
                "fill-mask",
                model=model,
                tokenizer=self.tokenizer,
                device=resolved_device,
                top_k=self.top_k
            )

            logger.info(f"Initializing internal DictionarySpellChecker (lang={dict_language}, dist={dict_distance}).")
            self.dict_checker = DictionarySpellChecker(
                language=dict_language,
                distance=dict_distance,
                custom_dictionary=dict_custom_dictionary
            )
            if not self.dict_checker.spell:
                 raise RuntimeError("Internal DictionarySpellChecker failed to initialize.")

            logger.info("ContextualLMSpellChecker initialized successfully.")

        except Exception as e:
            logger.error(f"Failed to initialize components for model '{model_name}': {e}", exc_info=True)
            self.pipeline = None
            self.tokenizer = None
            self.mask_token = None
            self.dict_checker = None

    def is_correct(self, word: str) -> bool:
        """
        Checks if the word is considered correct by the internal dictionary checker.
        """
        if not self.dict_checker:
            logger.warning("Internal dictionary checker not available, assuming word is correct.")
            return True
        return self.dict_checker.is_correct(word)

    def suggest(self, word: str) -> List[str]:
        """
        Suggests corrections by masking the word, getting LM predictions,
        and filtering them using the dictionary and Levenshtein distance.
        WARNING: Lacks context. Use correct_text for context-aware corrections.
        """
        if not self.pipeline or not self.mask_token or not self.dict_checker:
            logger.warning("ContextualLMSpellChecker not initialized properly, returning no suggestions.")
            return []

        try:
            results = self.pipeline(f"{self.mask_token}")

            valid_suggestions = []
            seen_suggestions = set()

            for r in results:
                suggestion = r['token_str'].strip()
                suggestion_lower = suggestion.lower()

                if not suggestion or not re.match(r'^\w+$', suggestion) or suggestion_lower in seen_suggestions:
                    continue

                if not self.dict_checker.is_correct(suggestion):
                    continue

                if self.suggestion_distance_threshold is not None and _levenshtein_available:
                    distance = Levenshtein.distance(word.lower(), suggestion_lower)
                    if distance > self.suggestion_distance_threshold:
                        continue

                valid_suggestions.append(suggestion)
                seen_suggestions.add(suggestion_lower)

            return valid_suggestions

        except Exception as e:
            logger.error(f"Error during suggestion generation for '{word}': {e}", exc_info=True)
            return []

    def correct_word(self, word: str) -> str:
        """
        Returns the most likely valid correction based on LM score, dictionary check, and distance.
        WARNING: Lacks context. Use correct_text for context-aware corrections.
        """
        suggestions = self.suggest(word)

        if suggestions:
             if suggestions[0].lower() != word.lower():
                 return suggestions[0]

        return word


    def correct_text(self, text: str) -> str:
        """
        Corrects text contextually by masking each word, predicting replacements,
        filtering suggestions (dictionary, Levenshtein), and selecting the best candidate.
        """
        if not self.pipeline or not self.tokenizer or not self.mask_token or not self.dict_checker:
            logger.warning("ContextualLMSpellChecker not initialized properly, returning original text.")
            return text

        tokens_info = self._simple_tokenizer(text)
        original_words = [token for token, is_word in tokens_info if is_word]

        if not original_words:
            return text

        corrected_word_list = list(original_words)

        for i, word_to_check in enumerate(original_words):
            word_to_check_lower = word_to_check.lower()

            if self.dict_checker.is_correct(word_to_check):
                 logger.debug(f"'{word_to_check}' is correct according to dictionary, skipping LM prediction.")
                 continue

            temp_tokens = list(original_words)
            temp_tokens[i] = self.mask_token
            masked_context = " ".join(temp_tokens)

            try:
                results = self.pipeline(masked_context)

                best_candidate: Optional[str] = None
                best_score: float = -float('inf')
                min_distance: int = self.suggestion_distance_threshold + 1 if self.suggestion_distance_threshold is not None else float('inf')


                if results:
                    for r in results:
                        suggestion = r['token_str'].strip()
                        score = r['score']
                        suggestion_lower = suggestion.lower()

                        if not suggestion or not re.match(r'^\w+$', suggestion):
                            continue

                        if not self.dict_checker.is_correct(suggestion):
                            continue

                        distance = 0
                        if self.suggestion_distance_threshold is not None and _levenshtein_available:
                            distance = Levenshtein.distance(word_to_check_lower, suggestion_lower)
                            if distance > self.suggestion_distance_threshold:
                                continue

                        if distance < min_distance:
                             min_distance = distance
                             best_score = score
                             best_candidate = suggestion
                        elif distance == min_distance and score > best_score:
                             best_score = score
                             best_candidate = suggestion

                if best_candidate and best_candidate.lower() != word_to_check_lower:
                    logger.debug(f"Correcting '{word_to_check}' to '{best_candidate}' (dist={min_distance}, score={best_score:.4f})")
                    corrected_word_list[i] = best_candidate
                elif best_candidate and best_candidate.lower() == word_to_check_lower:
                     logger.debug(f"Best candidate '{best_candidate}' is same as original '{word_to_check}', keeping original.")
                else:
                     logger.debug(f"No suitable correction found for '{word_to_check}' within constraints.")


            except Exception as e:
                 logger.warning(f"Error processing word '{word_to_check}' at index {i} in context: {e}", exc_info=False)
                 continue

        final_parts = []
        word_idx = 0
        for token, is_word in tokens_info:
            if is_word:
                corrected_word = corrected_word_list[word_idx]

                if token.istitle() and len(corrected_word) > 0:
                    corrected_word = corrected_word.capitalize()
                elif token.isupper():
                     if len(token) > 1 or len(corrected_word) == 1:
                          corrected_word = corrected_word.upper()
                     else:
                          corrected_word = corrected_word.upper()

                final_parts.append(corrected_word)
                word_idx += 1
            else:
                final_parts.append(token)

        return "".join(final_parts)

    def get_implementation_details(self) -> str:
         details = f"Using implementation: {self.__class__.__name__}\n"
         details += f"  LM Model: {self.model_name}\n"
         details += f"  LM Top-K: {self.top_k}\n"
         details += f"  Levenshtein Threshold: {self.suggestion_distance_threshold if _levenshtein_available else 'N/A (Levenshtein not installed)'}\n"
         if self.dict_checker and self.dict_checker.spell:
             details += f"  Internal Dictionary Lang: {self.dict_checker._language}\n"
             details += f"  Internal Dictionary Distance: {self.dict_checker.spell.distance}"
         else:
             details += "  Internal Dictionary: Not Initialized"
         return details
