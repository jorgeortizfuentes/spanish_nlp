import logging

from . import augmentation
from .augmentation import *
from .classifiers import SpanishClassifier
from .preprocess import SpanishPreprocess
from .spellchecker import SpanishSpellChecker

# Configure logging for the library to avoid 'No handler found' warnings
logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "SpanishClassifier",
    "SpanishPreprocess",
    "SpanishSpellChecker",
    "augmentation",
]
