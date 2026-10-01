import logging

from .augmentation import *
from .classifiers import SpanishClassifier
from .preprocess import SpanishPreprocess
from .spellchecker import SpanishSpellChecker

# Configure logging for the library to avoid 'No handler found' warnings
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

try:
    from .preprocess import SpanishPreprocess
except (ImportError, ModuleNotFoundError) as e:
    logger.error("Could not import SpanishPreprocess: %s", e)
    SpanishPreprocess = None

try:
    from .spellchecker import SpanishSpellChecker
except (ImportError, ModuleNotFoundError) as e:
    logger.debug("SpanishSpellChecker not available: %s", e)
    SpanishSpellChecker = None

try:
    from . import augmentation
except (ImportError, ModuleNotFoundError) as e:
    logger.debug("Augmentation module not available: %s", e)
    augmentation = None

try:
    from . import classifiers
except (ImportError, ModuleNotFoundError) as e:
    logger.debug("Classifiers module not available: %s", e)
    classifiers = None

__all__ = [
    "SpanishClassifier",
    "SpanishPreprocess",
    "SpanishSpellChecker",
    "augmentation",  # Or list specific classes like "Spelling", "Masked"
    "classifiers",
]
