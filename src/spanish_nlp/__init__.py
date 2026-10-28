import logging

from .preprocess import SpanishPreprocess

# Configure logging for the library to avoid 'No handler found' warnings
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


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
    from .classifiers import SpanishClassifier
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
