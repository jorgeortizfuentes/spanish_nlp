import re

import numpy as np
from math import ceil
from unidecode import unidecode
import json
import os

from .abstract import DataAugmentationAbstract
import random


class Spelling(DataAugmentationAbstract):
    """
    Class for data augmentation with spelling.
    """

    def __init__(
        self, method, stopwords="default", aug_percent=0.1, tokenizer="default"
    ):
        """
        Initialize the class.
        """
        super().__init__(method)

        available_methods = [
            "keyboard",
            "ocr",
            "random",
            "grapheme_spelling",
            "word_spelling",
            "remove_punctuation",
            "remove_accents",
            "remove_spaces",
            "lowercase",
            "uppercase",
            "randomcase",
            "all",
        ]

        if self.method not in available_methods:
            str_methods = ", ".join(available_methods)
            raise ValueError(
                f"Method not available. The method must be {str_methods}."
            )

        self.aug_percent = aug_percent
        self.tokenizer = tokenizer
        self.stopwords = stopwords

        if self.tokenizer == "default":
            self._load_default_tokenizer_()
        if self.stopwords == "default":
            self._load_default_stopwords_()

    def set_aug_percent(self, aug_percent):
        """
        Set the augmentation percentage.
        """
        if not isinstance(aug_percent, float):
            raise ValueError("The aug_percent must be a float.")
        self.aug_percent = aug_percent

    def set_tokenizer(self, tokenizer):
        """
        Set the tokenizer.
        """
        self.tokenizer = tokenizer

    def _text_augment_(self, text, num_samples=1):
        """
        Augment a single text.
        """
        if self.method == "keyboard":
            return self._keyboard_augment_(text, num_samples)
        elif self.method == "ocr":
            return self._ocr_augment_(text, num_samples)
        elif self.method == "random":
            return self._random_augment_(text, num_samples)
        elif self.method == "grapheme_spelling":
            return self._grapheme_spelling_augment_(text, num_samples)
        elif self.method == "word_spelling":
            return self._word_spelling_augmentation_(text, num_samples)
        elif self.method == "remove_punctuation":
            return self._remove_punctuation_augment_(text, num_samples)
        elif self.method == "remove_spaces":
            return self._remove_spaces_augment_(text, num_samples)
        elif self.method == "remove_accents":
            return self._remove_accents_augmentation_(text, num_samples)
        elif self.method == "lowercase":
            return self._lowercase_augmentation_(text, num_samples)
        elif self.method == "uppercase":
            return self._uppercase_augmentation_(text, num_samples)
        elif self.method == "randomcase":
            return self._random_case_augmentation_(text, num_samples)
        elif self.method == "all":
            return self._all_augment_(text, num_samples)
        else:
            raise ValueError("Invalid method")

    def _set_keyboard_augment_dict_(self):
        """Create keyboard augment dict for qwerty keyboard"""
        self.keyboard_augment_dict = {
            "q": ["w", "a", "s"],
            "w": ["q", "e", "a", "s", "d"],
            "e": ["w", "s", "d", "f", "r"],
            "r": ["e", "d", "f", "g", "t"],
            "t": ["r", "f", "g", "h", "y"],
            "y": ["t", "g", "h", "j", "u"],
            "u": ["y", "h", "j", "k", "i"],
            "i": ["u", "j", "k", "l", "o"],
            "o": ["i", "k", "l", "p"],
            "p": ["o", "l", "ñ"],
            "a": ["q", "w", "s", "z", "x"],
            "s": ["a", "q", "w", "e", "d", "z", "x"],
            "d": ["s", "w", "e", "r", "f", "x", "c"],
            "f": ["d", "e", "r", "t", "g", "c", "v"],
            "g": ["f", "r", "t", "y", "h", "v", "b"],
            "h": ["g", "t", "y", "u", "j", "b", "n"],
            "j": ["h", "y", "u", "i", "k", "n", "m"],
            "k": ["j", "u", "i", "o", "l", "m"],
            "l": ["k", "i", "o", "p", "ñ"],
            "ñ": ["l", "o", "p"],
            "z": ["a", "s", "x"],
            "x": ["z", "s", "d", "c"],
            "c": ["x", "d", "f", "v"],
            "v": ["c", "f", "g", "b"],
            "b": ["v", "g", "h", "n"],
            "n": ["b", "h", "j", "m"],
            "m": ["n", "j", "k"],
        }

    def _keyboard_augment_(self, text, num_samples):
        """
        Increase textual data by modifying characters randomly according to the proximity of the keystrokes
        """
        # If self.keyboard_augment_dict is not defined, define it
        if not hasattr(self, "keyboard_augment_dict"):
            self._set_keyboard_augment_dict_()

        # List to save the augmented texts
        output_texts = []

        # Iterate over the number of samples
        for i in range(num_samples):
            # Get the number of characters to augment
            num_aug = int(len(text) * self.aug_percent)

            # Get the indices of the characters to augment
            aug_indices = np.random.choice(
                range(len(text)), num_aug, replace=False
            )
            # Get the characters to augment
            aug_chars = [text[i] for i in aug_indices]
            # Get the augmented characters
            aug_chars = [
                self.keyboard_augment_dict[c][
                    np.random.randint(0, len(self.keyboard_augment_dict[c]))
                ]
                if c in self.keyboard_augment_dict
                else c
                for c in aug_chars
            ]
            # Augment the text
            output_text = list(text)
            for j in range(num_aug):
                output_text[aug_indices[j]] = aug_chars[j]
            output_text = "".join(output_text)
            # Append the augmented text to the list
            output_texts.append(output_text)

        return output_texts

    def _set_ocr_augment_dict_(self):
        self.ocr_augment_dict = {
            "0": ["D", "0", "o", "8"],
            "1": ["I", "l", "i", "L"],
            "2": ["Z", "z"],
            "3": ["E", "e"],
            "4": ["A", "a"],
            "5": ["S", "s"],
            "6": ["G", "g"],
            "7": ["T", "t"],
            "8": ["B", "s", "S"],
            "9": ["g", "q"],
            "A": ["4"],
            "B": ["8", "E"],
            "C": ["G"],
            "D": ["0", "O"],
            "E": ["3"],
            "F": ["T"],
            "G": ["6", "C"],
            "H": ["4"],
            "I": ["1", "l", "i", "L"],
            "J": ["L"],
            "K": ["X"],
            "L": ["1", "i", "I"],
            "M": ["W"],
            "N": ["H"],
            "Ñ": ["N", "n"],
            "O": ["0", "Q"],
            "P": ["R"],
            "Q": ["O", "9"],
            "R": ["P"],
            "S": ["5", "8", "B"],
            "T": ["7"],
            "U": ["V", "v"],
            "V": ["U"],
            "X": ["K"],
            "Y": ["V"],
            "Z": ["2", "Z"],
            "b": ["8", "E"],
            "c": ["G"],
            "d": ["0", "O"],
            "e": ["3"],
            "f": ["T"],
            "g": ["6", "C", "9", "q"],
            "h": ["4"],
            "i": ["1", "l", "I"],
            "j": ["L"],
            "k": ["X"],
            "l": ["1", "i", "I"],
            "m": ["W"],
            "ñ": ["N", "n"],
            "o": ["0", "Q"],
            "p": ["R"],
            "q": ["O", "9", "g"],
            "r": ["P"],
            "s": ["5", "8", "B"],
            "t": ["7"],
            "u": ["V"],
            "v": ["U"],
            "w": ["M", "V"],
            "x": ["K"],
            "y": ["V"],
            "z": ["2", "Z"],
        }

    def _ocr_augment_(self, text, num_samples):
        """
        Increase textual data by modifying characters randomly according to the common OCR errors for Spanish
        """
        # If self.keyboard_augment_dict is not defined, define it
        if not hasattr(self, "ocr_augment_dict"):
            self._set_ocr_augment_dict_()

        # List to save the augmented texts
        output_texts = []

        # Iterate over the number of samples
        for i in range(num_samples):
            # Get the number of characters to augment
            num_aug = int(len(text) * self.aug_percent)
            # Get the indices of the characters to augment
            aug_indices = np.random.choice(
                range(len(text)), num_aug, replace=False
            )

            # Get the characters to augment
            aug_chars = [text[i] for i in aug_indices]

            # Get the augmented characters
            aug_chars = [
                self.ocr_augment_dict[c][
                    np.random.randint(0, len(self.ocr_augment_dict[c]))
                ]
                if c in self.ocr_augment_dict
                else c
                for c in aug_chars
            ]
            aug_text = list(text)
            for i, c in zip(aug_indices, aug_chars):
                aug_text[i] = c
            aug_text = "".join(aug_text)
            # Add the augmented text to the list
            output_texts.append(aug_text)
        return output_texts

    def _set_alphabet_(self):
        """
        Set the alphabet of the dataset
        """
        alphabet = []
        alphabet.extend([chr(i) for i in range(ord("A"), ord("Z") + 1)])
        alphabet.extend([chr(i) for i in range(ord("a"), ord("z") + 1)])
        alphabet.extend([chr(i) for i in range(ord("0"), ord("9") + 1)])
        alphabet.extend(
            [
                "á",
                "é",
                "í",
                "ó",
                "ú",
                "ñ",
                "ü",
                "Á",
                "É",
                "Í",
                "Ó",
                "Ú",
                "Ñ",
                "Ü",
            ]
        )
        self.alphabet = alphabet

    def _random_augment_(self, text, num_samples):
        """
        Increase textual data by modifying characters randomly
        """
        if not hasattr(self, "alphabet"):
            self._set_alphabet_()

        # List to save the augmented texts
        output_texts = []

        # Iterate over the number of samples
        for i in range(num_samples):
            # Get the number of characters to augment
            num_aug = int(len(text) * self.aug_percent)
            # Get the indices of the characters to augment
            aug_indices = np.random.choice(
                range(len(text)), num_aug, replace=False
            )
            # Get the characters to augment
            aug_chars = [text[i] for i in aug_indices]
            # Get the augmented characters if they are in the alphabet
            aug_chars = [
                np.random.choice(self.alphabet) if c in self.alphabet else c
                for c in aug_chars
            ]
            # Augment the text
            output_text = list(text)
            for j in range(num_aug):
                output_text[aug_indices[j]] = aug_chars[j]

            output_text = "".join(output_text)
            # Append the augmented text to the list
            output_texts.append(output_text)
        return output_texts

    def _set_grapheme_spelling_dict_(self):
        """
        Set the grapheme_spelling dictionary
        """
        grapheme_spelling_dict = {
            "mb": "nv",
            "nv": "mb",
            "m": "n",
            "n": "m",
            "v": "b",
            "b": "v",
            "ll": "y汉",  # then replace "汉" to ""
            "y": "漢",  # then replace "漢" to "ll",
            "h": " ",
            "qu": "k汉",  # then replace "汉" to ""
            "g": "j",
            "j": "g",
            "z": "s",
            "ci": "si",
        }

        self.grapheme_spelling_dict = grapheme_spelling_dict

    def __find_substring_indexes__(self, string, substring):
        """Find all indexes of a substring in a string.

        Args:
            string (str): string to search in.
            substring (str): substring to search for.

        Returns:
            list: list of tuples with the start and end indexes of the substring.
        """
        pattern = re.compile(re.escape(substring))
        matches = pattern.finditer(string)
        return [(match.start(), match.end() - 1) for match in matches]

    def _grapheme_spelling_augment_(self, text, num_samples):
        """
        Increase textual data by modifying characters randomly according to the common grapheme_spellings for Spanish

        """
        # If self.grapheme_spelling_dict is not defined, define it
        if not hasattr(self, "grapheme_spelling_dict"):
            self._set_grapheme_spelling_dict_()
        # List to save the augmented texts
        output_texts = []

        # Count the times that are self.keyboard_augment_dict keys in the text with self.__find_substring_indexes__(string, substring)
        aparitions = [
            self.__find_substring_indexes__(text, key)
            for key in self.grapheme_spelling_dict.keys()
        ]
        # Join all the sublists in a single list
        # Get the number of characters to augment
        aparitions = sum(aparitions, [])
        num_aug = int(len(aparitions) * self.aug_percent)
        # Iterate over the number of samples
        for i in range(num_samples):
            new_text = text
            # Create a list with num_aug elements in aparitions without repetition
            elements = random.sample(aparitions, num_aug)
            for e in elements:
                start = e[0]
                end = e[1] + 1
                substring = new_text[start:end]
                replacement = self.grapheme_spelling_dict[substring]
                # Select a random element from the list of replacements
                # replacement = random.choice(replacement)
                start_str = new_text[:start]
                end_str = new_text[end:] if end < len(new_text) else ""
                new_text = start_str + replacement + end_str

            # Replace auxiliar characters
            new_text = new_text.replace("汉", "")
            new_text = new_text.replace("漢", "ll")
            # Append the augmented text to the list
            output_texts.append(new_text)
        return output_texts

    def _remove_punctuation_augment_(self, text, num_samples):
        """Remove punctuation in the text according to the aug_percent

        Args:
            text (str): text to augment
            num_samples (int): number of samples to generate
        """
        # List to save the augmented texts
        output_texts = []
        # List with the punctuation to eliminate
        punctuation = [".", ",", "¡", "¿"]
        num_text_punctuation = [c for c in text if c in punctuation]
        num_aug = len(num_text_punctuation) * self.aug_percent
        # Round by the upper integer
        num_aug = ceil(num_aug)
        for i in range(num_samples):
            new_text = text
            # Get the indices of the punctuation to eliminate
            indices = [i for i, c in enumerate(new_text) if c in punctuation]
            # Create a list with num_aug elements in indices without repetition
            elements = random.sample(indices, num_aug)
            for e in elements:
                new_text = new_text[:e] + new_text[e + 1 :]
            # Append the augmented text to the list
            output_texts.append(new_text)
        return output_texts

    def _remove_spaces_augment_(self, text, num_samples):
        """Remove spaces in the text according to the aug_percent

        Args:
            text (str): text to augment
            num_samples (int): number of samples to generate
        """
        # List to save the augmented texts
        output_texts = []
        # List with the punctuation to eliminate
        num_text_spaces = [c for c in text if c == " "]
        num_aug = len(num_text_spaces) * self.aug_percent
        # Round by the upper integer
        num_aug = ceil(num_aug)
        for i in range(num_samples):
            new_text = text
            # Get the indices of the punctuation to eliminate
            indices = [i for i, c in enumerate(new_text) if c == " "]
            # Create a list with num_aug elements in indices without repetition
            elements = random.sample(indices, num_aug)
            for e in elements:
                new_text = new_text[:e] + new_text[e + 1:]
            # Append the augmented text to the list
            output_texts.append(new_text)
        return output_texts

    def _remove_accents_augmentation_(self, text, num_samples):
        """Remove accents in the text according to the aug_percent

        Args:
            text (str): text to augment
            num_samples (int): number of samples to generate
        """
        # List to save the augmented texts
        output_texts = []
        # List with the punctuation to eliminate
        accents = ["á", "é", "í", "ó", "ú", "Á", "É", "Í", "Ó", "Ú", "ü", "Ü"]
        num_text_accents = [c for c in text if c in accents]
        num_aug = len(num_text_accents) * self.aug_percent
        # Round by the upper integer
        num_aug = ceil(num_aug)
        for i in range(num_samples):
            new_text = text
            # Get the indices of the punctuation to eliminate
            indices = [i for i, c in enumerate(new_text) if c in accents]
            # Create a list with num_aug elements in indices without repetition
            elements = random.sample(indices, num_aug)
            # Replace every element with accent with the same element without accent
            for e in elements:
                new_e = unidecode(new_text[e])
                new_text = new_text[:e] + new_e + new_text[e + 1 :]

            # Append the augmented text to the list
            output_texts.append(new_text)
        return output_texts

    def _lowercase_augmentation_(self, text, num_samples):
        """Randomly chooses words containing at least one capital letter
        and converts them to lowercase according to the aug_percent.

        Args:
            text (str): text to augment
            num_samples (int): number of samples to generate
        """
        # List to save the augmented texts
        output_texts = []
        uppercase_words = []
        word = ""
        start = 0
        # Add to uppercase_words the words (separated by spaces) with at least one letter uppercase ({'word': 'word', 'start': first_int, 'end': last_int}})
        for index, character in enumerate(text):
            if character != " ":
                word += character
            else:
                end = index
                if any(char.isupper() for char in word):
                    uppercase_words.append(
                        {
                            "word": word,
                            "start": index - len(word),
                            "end": index,
                        }
                    )
                start = index + 1
                word = ""

        # Num of words to lowercase
        num_aug = ceil(len(uppercase_words) * self.aug_percent)
        for i in range(num_samples):
            # Copy the original text
            new_text = text
            # Choose a random subset of words to lowercase
            selected_words = random.sample(uppercase_words, num_aug)
            # Lowercase the selected words in the new text
            for word in selected_words:
                new_word = word["word"].lower()
                new_text = (
                    new_text[: word["start"]]
                    + new_word
                    + new_text[word["end"] :]
                )
            output_texts.append(new_text)
        return output_texts

    def _uppercase_augmentation_(self, text, num_samples):
        """Randomly chooses words containing at least one lower letter
        and converts them to lowercase according to the aug_percent.

        Args:
            text (str): text to augment
            num_samples (int): number of samples to generate
        """
        # List to save the augmented texts
        output_texts = []
        lowercase_worsd = []
        word = ""
        start = 0
        # Add to lowercase_worsd the words (separated by spaces) with at least one letter lowercase ({'word': 'word', 'start': first_int, 'end': last_int}})
        for index, character in enumerate(text):
            if character != " ":
                word += character
            else:
                end = index
                if any(char.islower() for char in word):
                    lowercase_worsd.append(
                        {
                            "word": word,
                            "start": index - len(word),
                            "end": index,
                        }
                    )
                start = index + 1
                word = ""

        # Num of words to lowercase
        num_aug = ceil(len(lowercase_worsd) * self.aug_percent)
        for i in range(num_samples):
            # Copy the original text
            new_text = text
            # Choose a random subset of words to lowercase
            selected_words = random.sample(lowercase_worsd, num_aug)
            # Lowercase the selected words in the new text
            for word in selected_words:
                new_word = word["word"].upper()
                new_text = (
                    new_text[: word["start"]]
                    + new_word
                    + new_text[word["end"] :]
                )
            output_texts.append(new_text)
        return output_texts

    def _random_case_augmentation_(self, text, num_samples):
        """Randomly lowercase or uppercase words in the text according to the aug_percent

        Args:
            text (str): text to augment
            num_samples (int): number of samples to generate
        """
        # List to save the augmented texts
        output_texts = []
        half_aug_percent = self.aug_percent / 2

        for i in range(num_samples):
            # Copy the original text
            new_text = text
            # Lowercase the text
            new_text = new_text.lower()
            # Uppercase the text
            new_text = self._uppercase_augmentation_(new_text, 1)[0]
            # Lowercase the text
            new_text = self._lowercase_augmentation_(new_text, 1)[0]
            # Append the augmented text to the list
            output_texts.append(new_text)
        return output_texts

    def _load_mispelled_words_dictionary_(self):
        """
        Set the dictionary of misspelled words
        """
        # Load the dictionary
        json_path = os.path.join(
            os.path.dirname(__file__), "data", "misspelled_words.json"
        )
        with open(json_path, "r") as f:
            self.misspelled_dict = json.load(f)

    def _word_spelling_augmentation_(self, text, num_samples):
        """Randomly chooses words from a dictionary and replaces them with a random misspelled word

        Args:
            text (str): text to augment
            num_samples (int): number of samples to generate
        """
        # If self.misspelled_dict is not defined, define it
        if not hasattr(self, "misspelled_dict"):
            self._load_mispelled_words_dictionary_()
        output_texts = []
        # Split the text in words
        words = text.split(" ")
        # Check words in the dictionary
        words = [word for word in words if word in self.misspelled_dict]
        # Num of words to replace
        num_aug = ceil(len(words) * self.aug_percent)
        for i in range(num_samples):
            # Copy the original text
            new_text = text
            # Choose a random subset of words to replace
            selected_words = random.sample(words, num_aug)
            # Replace the selected words in the new text
            for word in selected_words:
                # Select a random misspelled word
                new_word = random.choice(self.misspelled_dict[word])
                new_text = new_text.replace(word, new_word)
            output_texts.append(new_text)
        return output_texts

    def _all_augment_(self, text, num_samples):
        """
        Increase textual data by modifying characters randomly according to the common grapheme_spellings for Spanish
        """
        aug_percent = self.aug_percent
        self.aug_percent = aug_percent / 7
        output_texts = []
        for i in range(num_samples):
            text = self._word_spelling_augmentation_(text, 1)
            text = self._grapheme_spelling_augment_(text[0], 1)
            text = self._keyboard_augment_(text[0], 1)
            text = self._ocr_augment_(text[0], 1)
            text = self._random_augment_(text[0], 1)
            text = self._remove_punctuation_augment_(text[0], 1)
            text = self._remove_spaces_augment_(text[0], 1)
            text = self._remove_accents_augmentation_(text[0], 1)
            text = self._random_case_augmentation_(text[0], 1)
            output_texts.append(text[0])
        self.aug_percent = aug_percent
        return output_texts
