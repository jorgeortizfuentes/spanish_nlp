import logging

from .augmentation import *
from .classifiers import SpanishClassifier
from .preprocess import SpanishPreprocess
from .spellchecker import SpanishSpellChecker

# Configure logging for the library to avoid 'No handler found' warnings
logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "SpanishPreprocess",
    "SpanishClassifier",
    "SpanishSpellChecker",
    # Re-exporting augmentation classes might be needed depending on usage
    "augmentation",  # Or list specific classes like "Spelling", "Masked"
]
