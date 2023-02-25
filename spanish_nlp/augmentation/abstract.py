import os

import es_core_news_sm
import pandas as pd
from datasets import load_dataset


class DataAugmentationAbstract:
    """
    Abstract class for data augmentation.
    """

    def __init__(self, method):
        """
        Initialize the class.
        """
        self.method = method

    def augment(self, texts, num_samples=1, num_workers=-1):
        """
        Augment the data.
        """
        if num_workers == -1:
            # Get the number of CPUs in the system
            num_workers = os.cpu_count() - 1
        if isinstance(texts, str):
            return self._text_augment_(texts, num_samples)
        elif isinstance(texts, list):
            return self._list_augment_(texts, num_samples, num_workers)
        elif isinstance(texts, pd.Series):
            return self._pandas_augment_(texts, num_samples, num_workers)
        elif isinstance(texts, load_dataset.Dataset):
            return self._datasets_augment_(texts, num_samples, num_workers)
        else:
            raise ValueError(
                "The texts must be a string, a list of strings or a pandas Series."
            )

    def _text_augment_(self, text, num_samples):
        """
        Augment a single text.
        """
        raise NotImplementedError("This method must be implemented in the child class.")

    def _list_augment_(self, texts, num_samples, num_workers):
        """
        Augment a list of texts.
        """
        return self._pandas_augment_(
            pd.Series(texts), num_samples, num_workers
        ).tolist()

    def _pandas_augment_(self, texts, num_samples, num_workers):
        """
        Augment a pandas Series.
        """
        return texts.swifter.apply(
            self._text_augment_, num_samples=num_samples, num_workers=num_workers
        )

    def _datasets_augment_(self, dataset, num_samples, num_workers):
        """
        Augment a datasets Dataset.
        """
        # Not Implemented yet, return error
        raise NotImplementedError("This method is not implemented yet.")

    def _load_default_tokenizer_(self):
        # import es_core_news_sm if it is not imported
        # if "es_core_news_sm" not in sys.modules:
        #     import es_core_news_sm

        if not hasattr(self, "nlp_spacy"):
            self.nlp_spacy = es_core_news_sm.load(
                disable=["ner", "parser", "tagger", "textcat", "vectors"]
            )

        def tokenizer(text):
            return [token.text for token in self.nlp_spacy(text)]

        self.tokenizer = tokenizer

    def _load_default_stopwords_(self):
        from spanish_nlp.utils.stopwords import extended_stopwords

        self.stopwords = extended_stopwords
