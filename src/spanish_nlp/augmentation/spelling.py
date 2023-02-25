import re

import numpy as np

from .abstract import DataAugmentationAbstract


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

        if self.method not in [
            "keyboard",
            "ocr",
            "random",
            "misspelling",
            "all",
        ]:
            raise ValueError(
                "The method must be 'keyboard', 'ocr', 'random','misspelling' or 'all'"
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
        elif self.method == "misspelling":
            return self._misspelling_augment_(text, num_samples)
        elif self.method == "all":
            return self._all_augment_(text, num_samples)
        else:
            raise ValueError(
                "The method must be 'keyboard', 'ocr', 'random','misspelling' or 'all'"
            )

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
            aug_indices = np.random.choice(range(len(text)), num_aug, replace=False)
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
            aug_indices = np.random.choice(range(len(text)), num_aug, replace=False)

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
            aug_indices = np.random.choice(range(len(text)), num_aug, replace=False)
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

    def _set_misspelling_dict_(self):
        """
        Set the misspelling dictionary
        """
        misspelling_dict = {
            "mb": "nv",
            "nv": "mb",
            "m": "n",
            "n": "m",
            "v": "b",
            "b": "v",
            "ll": "y",
            "y": "ll",
            "h": "",
            "qu": "k",
            "g": "j",
            "j": "g",
            "z": "s",
            "ci": "si",
        }

        self.misspelling_dict = misspelling_dict

    def _misspelling_augment_(self, text, num_samples):
        """
        Increase textual data by modifying characters randomly according to the common misspellings for Spanish

        TODO: improve this function because only replace the first occurrence of the token
        """
        # If self.keyboard_augment_dict is not defined, define it
        if not hasattr(self, "misspelling_dict"):
            self._set_misspelling_dict_()

        output_texts = []
        # Tokenize the text
        tokens = self.tokenizer(text)
        num_aug = int(len(tokens) * self.aug_percent)
        new_text = text

        keys_token = list(self.misspelling_dict.keys())
        keys_token_in_text = [token for token in keys_token if token in text]

        for i in range(num_samples):
            n = num_aug
            new_text = text
            # Get a list of random number (without order and without
            # repetition). Numbers are int between 0 and
            # len(keys_token_in_text)-1
            random_numbers = np.random.choice(
                range(len(keys_token_in_text)), num_aug, replace=False
            )
            for rn in random_numbers:
                # Count keys_token_in_text[rn] in text
                n_rn = re.findall(keys_token_in_text[rn], text)
                if len(n_rn) > 0:
                    n -= len(n_rn)
                new_text = new_text.replace(
                    keys_token_in_text[rn],
                    self.misspelling_dict[keys_token_in_text[rn]],
                    2,
                )
                if n <= 0:
                    break
            output_texts.append(new_text)
        return output_texts

    def _all_augment_(self, text, num_samples):
        """
        Increase textual data by modifying characters randomly according to the common misspellings for Spanish
        """
        aug_percent = self.aug_percent
        self.aug_percent = aug_percent / 4
        output_texts = []
        for i in range(num_samples):
            text = self._misspelling_augment_(text, num_samples)
            text = self._keyboard_augment_(text, num_samples)
            text = self._ocr_augment_(text, num_samples)
            text = self._random_augment_(text, num_samples)
            output_texts.append(text)
        self.aug_percent = aug_percent
        return output_texts
