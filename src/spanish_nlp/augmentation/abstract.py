"""

Abstract class for data augmentation.

It was inspired by nlpaug (https://nlpaug.readthedocs.io/en/latest/augmenter/word/spelling.html)

These class allow to augmentate data with different ways: with spelling, language generative models, masked language models, back translation augmenter, word embeddings, abstractive summarization and synonyms.

TO DO:
* DataAugmentationSynonyms
** __init__(method, stopwords, aug_percent, dictionary, top_k)
* DataAugmentationWordEmbeddings
** __init__(method, stopwords, aug_percent, model, device, top_k)
* DataAugmentationGenerativeOpenSource
** __init__(method, stopwords, max_words, model, device, temperature, top_k, top_p, device, frequency_penalty, presence_penalty)
* DataAugmentationGenerativeOpenAI
** __init__(method, stopwords, max_words, model, token, temperature, top_k, top_p, frequency_penalty, presence_penalty)
* DataAugmentationBackTranslation
** __init__(method, stopwords, max_words, model, device, temperature, top_k)
* DataAugmentationAbstractiveSummarization
** __init__(method, stopwords, max_words, model, device, temperature, top_k)

DataAugmentationSynonyms uses WordNet to find synonyms.
DataAugmentationWordEmbeddings uses GenSim to find similar words (word2vec or fasttext)
Masked, DataAugmentationGenerative, DataAugmentationBackTranslation and DataAugmentationAbstractiveSummarization use HuggingFace transformers.

"""

import os

import es_core_news_sm
import pandas as pd
import swifter

from datasets import load_dataset
from tqdm import tqdm
tqdm.pandas()

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

    def _text_augment_(self, text, num_sample):
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
        # return texts.progress_apply(
        #     self._text_augment_,
        # )
        if num_workers == 1:
            return texts.progress_apply(
                self._text_augment_,
                num_samples=num_samples
            )
        else:
            return texts.swifter.apply(
                self._text_augment_,
                num_samples=num_samples,
            )

    def _datasets_augment_(self, dataset, num_samples, num_workers):
        """
        Augment a datasets Dataset.
        """
        # Apply self._text_augment_(text) to every text in dataset["text"] (HuggingFace datasets)
        dataset = dataset.map(
            self._text_augment_,
            batched=True,
            num_proc=num_workers,
            load_from_cache_file=False,
        )
        return dataset


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

    def set_stopwords(self, stopwords):
        """
        Set the stopwords.
        """
        self.stopwords = stopwords
