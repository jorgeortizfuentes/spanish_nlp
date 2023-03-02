import numpy as np
import math
import torch
from transformers import pipeline, AutoTokenizer
import warnings
from spanish_nlp.utils.stopwords import punct
from .abstract import DataAugmentationAbstract
import re

from transformers.utils import logging
logging.set_verbosity(40)


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
        if self.tokenizer == "default":
            self.tokenizer = model
        self.__load__tokenizer__()
        self.fillmask = pipeline(
            "fill-mask", model=model, tokenizer=self.tokenizer, device=self.device
        )
        self.mask_token = self.fillmask.tokenizer.mask_token

    def __load__tokenizer__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer,
                                                       truncation=True,
                                                       use_fast=True)

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
            # Tokenize text, count the tokens and if the tokens > max_length, return the original sentence
            n_tokens = len(self.tokenizer.tokenize(text))
            n_tokens_split = int(
                self.tokenizer.model_max_length / (self.aug_percent+1))+1
            n_splits = math.ceil((n_tokens / n_tokens_split))

            if n_splits > 1:
                new_text = self._large_sustitute_(text)
            else:
                new_text = self._normal_sustitute_(text)

            if new_text not in output_texts:
                output_texts.append(new_text)
            else:
                num_samples += 1

        return output_texts

    def _normal_sustitute_(self, sentence, num_samples=1):
        """
        Augments the input sentence by replacing a percentage of its non-stopword words with predicted words obtained by
        masking them using a pre-trained language model.

        Args:
            sentence (str): Input sentence to be augmented.
        Returns:
            str: Augmented sentence.
        """
        words = sentence.split(" ")
        not_allowed = punct + self.stopwords

        n_stopwords = sum([1 for word in words if word in not_allowed])
        n_total_words = len(words) - n_stopwords

        num_words = int(n_total_words * self.aug_percent)
        K_list = []

        # Select the words to be replaced
        while len(K_list) < num_words:
            K = np.random.randint(0, len(words))
            if words[K] not in punct:
                K_list.append(K)

        # Iterate over the words to be replaced
        for K in K_list:
            words = sentence.split(" ")
            masked_sentence = " ".join(
                words[:K] + [self.mask_token] + words[K + 1:])
            predictions = self.fillmask(masked_sentence, top_k=self.top_k)
            random_number = np.random.randint(0, self.top_k)
            new_word = predictions[random_number]["token_str"]

            # Verify that the new word is not a punctuation or a stopword
            count = 0
            while True:
                random_number = np.random.randint(0, self.top_k)
                # Check if there is a punctuation in new_word
                pattern = "|".join(re.escape(p) for p in punct)
                if re.search(pattern, new_word) == None and new_word not in not_allowed:
                    break
                elif count > self.top_k:
                    break
                else:
                    count += 1
            # Save the new sentence
            sentence = predictions[random_number]["sequence"]
        return sentence

    def _large_sustitute_(self, sentence, num_samples=1):
        """
        Augments the input sentence by replacing a percentage of its non-stopword words with predicted words obtained by
        masking them using a pre-trained language model. This method is used for long texts that exceed the maximum token
        length allowed by the language model.

        Args:
            sentence (str): Input sentence to be augmented.
        Returns:
            str: Augmented sentence.
        """
        # Split the text into chunks with the desired length
        n_tokens = len(self.tokenizer.tokenize(sentence))
        n_tokens_split = int(
            self.tokenizer.model_max_length / (self.aug_percent+1)) + 1
        n_splits = math.ceil((n_tokens / n_tokens_split))
        sentence_splits = self._split_text_into_chunks_(
            sentence, n_splits=n_splits)

        # Augment each chunk of the text and concatenate the results
        augmented_chunks = []
        for chunk in sentence_splits:
            augmented_chunk = self._normal_sustitute_(chunk, num_samples=1)
            augmented_chunks.append(augmented_chunk)

        augmented_sentence = " ".join(augmented_chunks)

        return augmented_sentence

    def _split_text_into_chunks_(self, text, n_splits):
        """
        Splits the input text into n_splits chunks with approximately the same length.

        Args:
            text (str): Input text to be split.
            n_splits (int): Number of chunks to split the text into.

        Returns:
            list of str: List of chunks.
        """
        tokens = self.tokenizer.tokenize(text)
        n_tokens = len(tokens)
        chunk_size = int(n_tokens / n_splits)
        chunks = [tokens[i:i+chunk_size]
                  for i in range(0, n_tokens, chunk_size)]
        return [" ".join(chunk) for chunk in chunks]

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
            # Tokenize text, count the tokens and if the tokens > max_length, return the original sentence
            n_tokens = len(self.tokenizer.tokenize(text))
            n_tokens_split = int(
                self.tokenizer.model_max_length / (self.aug_percent+1))+1
            n_splits = math.ceil((n_tokens / n_tokens_split))

            if n_splits > 1:
                new_text = self._large_insert_(text)
            else:
                new_text = self._normal_insert_(text)

            if new_text not in output_texts:
                output_texts.append(new_text)
            else:
                num_samples += 1
        return output_texts

    def _normal_insert_(self, sentence, num_samples=1):
        """Insert a self.aug_percent% of words within the sentence (not after punctuation marks)

        Args:
            sentence (str): Input sentence to be augmented.
        Returns:
            str: Augmented sentence.
        """

        # Tokens that are not allowed to be replaced
        words = sentence.split(" ")
        not_allowed = punct + self.stopwords
        # Tokenize text, count the tokens and if the tokens > max_length, return the original sentence
        tokens = len(self.tokenizer.tokenize(sentence))
        num_words = int(len(words) * self.aug_percent)
        K_list = []

        # Select the words to be replaced
        while len(K_list) < num_words:
            K = np.random.randint(0, len(words))
            if words[K] not in not_allowed:
                K_list.append(K)

        # Iterate over the words to be replaced
        for K in K_list:
            words = sentence.split(" ")
            masked_sentence = " ".join(
                words[:K] + [self.mask_token] + words[K:])
            predictions = self.fillmask(masked_sentence,
                                        top_k=self.top_k)
            random_number = np.random.randint(0, self.top_k)
            new_word = predictions[random_number]["token_str"]

            # Verify that the new word is not a punctuation or a stopword
            count = 0
            while True:
                random_number = np.random.randint(0, self.top_k)
                pattern = "|".join(re.escape(p) for p in punct)
                if re.search(pattern, new_word) == None and new_word not in not_allowed:
                    break
                elif count > self.top_k:
                    break
                else:
                    count += 1
            # Save the new sentence
            sentence = predictions[random_number]["sequence"]
        return sentence

    def _large_insert_(self, sentence, num_samples=1):
        """
        Augment a text with the insert method. This method inserts new words to the text.

        Args:
            sentence (str): Input sentence to be augmented.
        Returns:
            str: Augmented sentence.
        """
        # Split the text into chunks with the desired length
        n_tokens = len(self.tokenizer.tokenize(sentence))
        n_tokens_split = int(
            self.tokenizer.model_max_length / (self.aug_percent+1)) + 1
        n_splits = math.ceil((n_tokens / n_tokens_split))
        sentence_splits = self._split_text_into_chunks_(
            sentence, n_splits=n_splits)

        # Augment each chunk of the text and concatenate the results
        augmented_chunks = []
        for chunk in sentence_splits:
            augmented_chunk = self._normal_insert_(chunk, num_samples=1)
            augmented_chunks.append(augmented_chunk)

        augmented_sentence = " ".join(augmented_chunks)
        return augmented_sentence