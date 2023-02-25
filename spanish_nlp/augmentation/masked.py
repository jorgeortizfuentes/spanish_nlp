import numpy as np
import torch
from transformers import pipeline

from .abstract import DataAugmentationAbstract


class Masked(DataAugmentationAbstract):
    """
    Class for data augmentation with spelling. It uses the Huggingface pipeline for masked language modeling.

    """

    def __init__(
        self,
        method="sustitute",
        model="dccuchile/bert-base-spanish-wwm-cased",
        tokenizer="default",
        stopwords="default",
        aug_percent=0.4,
        device=None,
        top_k=10,
    ):
        """Init class.

        Args:
            model (str, optional): huggingface model. Defaults to "dccuchile/bert-base-spanish-wwm-cased".
            tokenizer (str, optional): huggingface tokenizer. Defaults to "default". If "default" is used, the model is used as tokenizer.
            stopwords (list or str, optional): list of stopwords. Defaults to "default". If "default" is used, the default stopwords are used.
            aug_percent (float, optional): percentage of words to be augmented. Defaults to 0.1.
            device (str or torch.device, optional): device to use. Defaults to None. If None is used, the device is automatically selected.
            top_k (int, optional): number of top k words to be used for augmentation. Defaults to 10.
        """
        super().__init__(method)

        if self.method not in ["sustitute", "insert"]:
            raise ValueError("The method must be 'sustitute' or 'insert'")

        # Set the parameters
        self.aug_percent = aug_percent
        self.tokenizer = tokenizer
        self.stopwords = stopwords
        self.top_k = top_k

        if self.stopwords == "default":
            self._load_default_stopwords_()

        # Download Huggingface model, tokenizer, device and mask token
        if device is None:
            # All available gpu devices if any else cpu
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device
        if tokenizer == "default":
            tokenizer = model
        self.fillmask = pipeline(
            "fill-mask", model=model, tokenizer=tokenizer, device=self.device
        )
        self.mask_token = self.fillmask.tokenizer.mask_token

    def set_aug_percent(self, aug_percent):
        """
        Set the augmentation percentage.
        """
        if not isinstance(aug_percent, float):
            raise ValueError("The aug_percent must be a float.")
        self.aug_percent = aug_percent

    def _text_augment_(self, text, num_samples=1):
        """
        Augment a single text.
        """
        if self.method == "sustitute":
            return self._sustitute_augment_(text, num_samples)
        elif self.method == "insert":
            return self._insert_augment_(text, num_samples)
        else:
            raise ValueError("The method must be 'sustitute' or 'insert'")

    def _augment_with_replace_(self, sentence):
        """
        Augments the input sentence by replacing a percentage of its non-stopword words with predicted words obtained by
        masking them using a pre-trained language model.

        Args:
            sentence (str): Input sentence to be augmented.
        Returns:
            str: Augmented sentence.
        """
        words = sentence.split(" ")
        punct = [
            ".",
            ",",
            ":",
            ";",
            "!",
            "?",
            "¿",
            "¡",
            "(",
            ")",
            "[",
            "]",
            "{",
            "}",
            "/",
            "\\",
            "|",
            "-",
            "_",
            "–",
            "—",
            "…",
            "·",
            "•",
            "°",
            "´",
            "`",
            "'",
            '"',
            "“",
            "”",
            "‘",
            "’",
            "«",
            "»",
            "‹",
            "›",
        ]
        not_allowed = punct + self.stopwords

        n_stopwords = sum([1 for word in words if word in not_allowed])
        n_total_words = len(words) - n_stopwords

        num_words = int(n_total_words * self.aug_percent)
        K_list = []

        # Select the words to be replaced
        while len(K_list) < num_words:
            K = np.random.randint(0, len(words))
            if words[K] not in not_allowed:
                K_list.append(K)

        # Iterate over the words to be replaced
        for K in K_list:
            words = sentence.split(" ")
            masked_sentence = " ".join(words[:K] + [self.mask_token] + words[K + 1 :])
            predictions = self.fillmask(masked_sentence, top_k=self.top_k)
            random_number = np.random.randint(0, self.top_k)
            new_word = predictions[random_number]["token_str"]

            # Verify that the new word is not a punctuation or a stopword
            count = 0
            while True:
                random_number = np.random.randint(0, self.top_k)
                if new_word not in not_allowed:
                    break
                elif count > self.top_k:
                    break
                else:
                    count += 1

            # Save the new sentence
            sentence = predictions[random_number]["sequence"]
        return sentence

    def _sustitute_augment_(self, text, num_samples=1):
        """
        Augment a text with the sustitute method. This method replaces a percentage of the words with other words.

        Args:
            text (str): text to be augmented.
            num_samples (int, optional): number of samples to be generated. Defaults to 1.

        Returns:
            list: list of augmented texts.
        """

        output_texts = []
        for _ in range(num_samples):
            new_text = self._augment_with_replace_(text)
            if new_text not in output_texts:
                output_texts.append(new_text)
            else:
                num_samples += 1

        return output_texts

    def _augment_with_insert_(self, sentence, max_words=450):
        """Insert a self.aug_percent% of words within the sentence (not after punctuation marks)

        Args:
            sentence (str): Input sentence to be augmented.
        Returns:
            str: Augmented sentence.
        """
        words = sentence.split(" ")
        punct = [
            ".",
            ",",
            ":",
            ";",
            "!",
            "?",
            "¿",
            "¡",
            "(",
            ")",
            "[",
            "]",
            "{",
            "}",
            "/",
            "\\",
            "|",
            "-",
            "_",
            "–",
            "—",
            "…",
            "·",
            "•",
            "°",
            "´",
            "`",
            "'",
            '"',
            "“",
            "”",
            "‘",
            "’",
            "«",
            "»",
            "‹",
            "›",
        ]
        not_allowed = punct + self.stopwords

        # If the sentence is too long, return it
        if len(words) > max_words:
            return sentence

        n_stopwords = sum([1 for word in words if word in not_allowed])
        n_total_words = len(words) - n_stopwords

        num_words = int(n_total_words * self.aug_percent)
        K_list = []

        # Select the words to be replaced
        while len(K_list) < num_words:
            K = np.random.randint(0, len(words))
            if words[K] not in not_allowed:
                K_list.append(K)

        # Iterate over the words to be replaced
        for K in K_list:
            words = sentence.split(" ")
            masked_sentence = " ".join(words[:K] + [self.mask_token] + words[K:])
            predictions = self.fillmask(masked_sentence, top_k=self.top_k)
            random_number = np.random.randint(0, self.top_k)
            new_word = predictions[random_number]["token_str"]

            # Verify that the new word is not a punctuation or a stopword
            count = 0
            while True:
                random_number = np.random.randint(0, self.top_k)
                if new_word not in not_allowed:
                    break
                elif count > self.top_k:
                    break
                else:
                    count += 1

            # Save the new sentence
            sentence = predictions[random_number]["sequence"]

            # If the sentence is too long, stop the augmentation and return the sentence
            if len(sentence.split(" ")) > max_words:
                return sentence
        return sentence

    def _insert_augment_(self, text, num_samples=1, max_words=450):
        """
        Augment a text with the insert method. This method inserts new words in the text.

        Args:
            text (str): text to be augmented.
            num_samples (int, optional): number of samples to be generated. Defaults to 1.

        Returns:
            list: list of augmented texts.
        """
        output_texts = []
        for _ in range(num_samples):
            new_text = self._augment_with_insert_(text, max_words=max_words)
            output_texts.append(new_text)

        return output_texts
