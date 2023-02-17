"""
# https://nlpaug.readthedocs.io/en/latest/augmenter/word/spelling.html

Classes Data Augmentation. The goal is to use it for text data augmentation. 

These class allow to augmentate data with different ways: with spelling, language generative models, masked language models, back translation augmenter, word embeddings, abstractive summarization and synonyms.  

Using object-oriented programming with an abstract class called Data Augmentation

* DataAugmentationAbstract:
* __init__(method, stopwords): method is a string, stopwords is a list of strings
* augment(texts, num_samples, num_workers): text is an string, a list of strings or a pandas Series and n_samples is an integer, and num_workers is an integer.
* _text_augment_(text, num_samples): text is a string and num_samples is an integer.
* _list_augment_(texts, num_samples): texts is a list of strings and num_samples is an integer.
* _pandas_augment_(texts, num_samples): texts is a pandas Series and num_samples is an integer. 

Then create the following classes that inherit from DataAugmentationAbstract:

* DataAugmentationSpelling
** __init__(method, stopwords, aug_percent, tokenizer): method is a string ("keyboard", "ocr", "random", "misspelling"),  min_words is an integer or a float and max_words is an integer or a float.
* DataAugmentationSynonyms
** __init__(method, stopwords, aug_percent, dictionary, top_k)
* DataAugmentationWordEmbeddings
** __init__(method, stopwords, aug_percent, model, device, top_k)
* DataAugmentationMasked
** __init__(method, stopwords, max_words, model, device, top_k)
* DataAugmentationGenerative
** __init__(method, stopwords, max_words, model, device, temperature, top_k, top_p, device, frequency_penalty, presence_penalty)
* DataAugmentationBackTranslation
** __init__(method, stopwords, max_words, model, device, temperature, top_k)
* DataAugmentationAbstractiveSummarization
** __init__(method, stopwords, max_words, model, device, temperature, top_k)

DataAugmentationSynonyms uses WordNet to find synonyms.
DataAugmentationWordEmbeddings uses GenSim to find similar words (word2vec or fasttext)
DataAugmentationMasked, DataAugmentationGenerative, DataAugmentationBackTranslation and DataAugmentationAbstractiveSummarization use HuggingFace transformers.

"""

class DataAugmentationAbstract:
    """
    Abstract class for data augmentation.
    """
    def __init__(self, method, stopwords):
        """
        Initialize the class.
        """
        self.method = method
        self.stopwords = stopwords

    def augment(self, texts, num_samples, num_workers):
        """
        Augment the data.
        """
        if isinstance(texts, str):
            return self._text_augment_(texts, num_samples)
        elif isinstance(texts, list):
            return self._list_augment_(texts, num_samples, num_workers)
        elif isinstance(texts, pd.Series):
            return self._pandas_augment_(texts, num_samples, num_workers)
        else:
            raise ValueError("The texts must be a string, a list of strings or a pandas Series.")

    def _text_augment_(self, text, num_samples):
        """
        Augment a single text.
        """
        raise NotImplementedError("This method must be implemented in the child class.")

    def _list_augment_(self, texts, num_samples, num_workers):
        """
        Augment a list of texts.
        """
        raise NotImplementedError("This method must be implemented in the child class.")

    def _pandas_augment_(self, texts, num_samples, num_workers):
        """
        Augment a pandas Series.
        """
        raise NotImplementedError("This method must be implemented in the child class.")

class DataAugmentationSpelling(DataAugmentationAbstract):
    """
    Class for data augmentation with spelling.
    """
    def __init__(self, method, stopwords, aug_percent, tokenizer):
        """
        Initialize the class.
        """
        super().__init__(method, stopwords)
        self.aug_percent = aug_percent
        self.tokenizer = tokenizer

    def _text_augment_(self, text, num_samples):
        """
        Augment a single text.
        """
        if self.method == "keyboard":
            return self._keyboard_augment_(text, num_samples)
        elif self.method == "ocr":
            return self._ocr_augment_(text, num_samples)
        elif self.method == "random":
            return self._random_augment_(text, num_samples)
        elif self.method == "misspelling":
            return self._misspelling_augment_(text, num_samples)
        else:
            raise ValueError("The method must be 'keyboard', 'ocr', 'random' or 'misspelling'.")

    def _list_augment_(self, texts, num_samples, num_workers):
        """
        Augment a list of texts.
        """
        if self.method == "keyboard":
            return self._keyboard_augment_(texts, num_samples, num_workers)
        elif self.method == "ocr":
            return self._ocr_augment_(texts, num_samples, num_workers)
        elif self.method == "random":
            return self._random_augment_(texts, num_samples, num_workers)
        elif self.method == "misspelling":
            return self._misspelling_augment_(texts, num_samples, num_workers)
        else:
            raise ValueError("The method must be 'keyboard', 'ocr', 'random' or 'misspelling'.")

    def _pandas_augment_(self, texts, num_samples, num_workers):
        """
        Augment a pandas Series.
        """
        if self.method == "keyboard":
            return self._keyboard_augment_(texts, num_samples, num_workers)
        elif self.method == "ocr":
            return self._ocr_augment_(texts, num_samples, num_workers)
        elif self.method == "random":
            return self._random_augment_(texts, num_samples, num_workers)
        elif self.method == "misspelling":
            return self._misspelling_augment_(texts, num_samples, num_workers)
        else:
            raise ValueError("The method must be 'keyboard', 'ocr', 'random' or 'misspelling'.")

    def _keyboard_augment_(self, texts, num_samples, num_workers):
        """
        Augment the data with keyboard.
        """
        if isinstance(texts, str):
            return self._keyboard_augment_text_(texts, num_samples)
        elif isinstance(texts, list):
            return self._keyboard_augment_list_(texts, num_samples, num_workers)
        elif isinstance(texts, pd.Series):
            return self._keyboard_augment_pandas_(texts, num_samples, num_workers)
        else:
            raise ValueError("The texts must be a string, a list of strings or a pandas Series.")

    def _keyboard_augment_text_(self, text, num_samples):
        """
        Augment a single text with keyboard.
        """
        augmented_texts = []
        for i in range(num_samples):
            augmented_text = text
            for j in range(len(text)):
                if random.random() < self.aug_percent:
                    augmented_text = augmented_text[:j] + random.choice(KEYBOARD_DISTANCE[text[j]]) + augmented_text[j+1:]
            augmented_texts.append(augmented_text)
        return augmented_texts

    def _keyboard_augment_list_(self, texts, num_samples, num_workers):
        """
        Augment a list of texts with keyboard.
        """
        augmented_texts = []
        with Pool(num_workers) as p:
            augmented_texts = p.starmap(self._keyboard_augment_text_, [(text