import re
import string

import emoji
from nltk.corpus import stopwords

from utilities.emo_unicode import demoticonize, emoticonize


class SpanishPreprocess:
    """Authors: Hernán Sarmiento, Ricardo Córdova y Jorge Ortiz"""

    def __init__(
        self,
        lower=True,
        remove_url=True,
        remove_hashtags=True,
        preserve_emoticons=True,
        preserve_emojis=True,
        convert_emoticons=False,
        convert_emojis=False,
        normalize_inclusive_language=True,
        reduce_spam=True,
        remove_vowels_accents=True,
        remove_punctuation=True,
        remove_unprintable=True,
        remove_numbers=True,
        remove_stopwords=True,
        stopwords_list=None,
        stem=True,
    ):
        self.lower = lower
        self.remove_url = remove_url
        self.remove_hashtags = remove_hashtags
        self.preserve_emojis = preserve_emojis
        self.preserve_emoticons = preserve_emoticons
        if self.preserve_emoticons:
            self.convert_emoticons = True
        else:
            self.convert_emoticons = convert_emoticons
        if self.preserve_emojis:
            self.convert_emojis = True
        else:
            self.convert_emojis = convert_emojis
        self.normalize_inclusive_language = normalize_inclusive_language
        self.reduce_spam = reduce_spam
        self.remove_vowels_accents = remove_vowels_accents
        self.remove_punctuation = remove_punctuation
        self.remove_numbers = remove_numbers
        self.remove_stopwords = remove_stopwords
        self.stopwords_list = stopwords_list
        self.remove_unprintable = remove_unprintable
        self.stem = stem

    def _lower_(self, text):
        return text.lower()

    def _remove_url_(self, text):
        text = re.sub(r"(?:\@|https?\://)\S+", "", text)
        url_pattern = re.compile(r"https?://\S+|www\.\S+")
        return url_pattern.sub(r"", text)

    def _remove_hashtags_(self, text):
        # text = re.sub("#[A-Za-z_Ññ_0-9_]+"," ", text)
        # return ' '.join(text.split())
        return text.replace(" #", " ").replace("#", " ")

    def _convert_emoticons_(self, text):
        return emoticonize(text, delimiters=(" __", "__ "))

    def _convert_emojis_(self, text):
        text = emoji.demojize(text, delimiters=(" __", "__ "))
        return text
        # return ' '.join(text.split())

    def _preserve_emoticons_(self, text):
        return demoticonize(text, delimiters=("__", "__"))

    def _preserve_emojis_(self, text):
        return emoji.emojize(text, delimiters=("__", "__"))

    def _normalize_inclusive_language_(self, text, inclusive_character="x"):
        text = re.sub(r"(@s)\b", rf"{inclusive_character}s", text)
        return re.sub(r"@([^a-z])", rf"{inclusive_character}\1", text)

    def _reduce_spam_(self, text):
        text = re.sub(r"(\w+\s)\1+", r"\1\1", text)
        text = re.sub(r"(\b(\w+\s){3})\1+", r"\1\1", text)
        return text

    def _remove_vowels_accents_(self, text):
        return (
            text.replace("á", "a")
            .replace("é", "e")
            .replace("í", "i")
            .replace("ó", "o")
            .replace("ú", "u")
            .replace("ü", "u")
        )

    def _remove_punctuation_(self, text):
        pattern = re.compile(r"[^\w\sáéíóúüñÁÉÍÓÚÜÑ]")
        t = pattern.sub(r" ", text)
        return re.sub(" +", " ", t)

    def _remove_unprintable_(self, text):
        printable = set(string.printable + "ñáéíóúü" + "ÑÁÉÍÓÚÜ")
        text = "".join(filter(lambda x: x in printable, text))
        return text

    def _remove_numbers_(self, text):
        text = re.sub(r"[0-9]", " ", text)
        return " ".join(text.split())

    def _remove_stopwords_(self, text):
        return " ".join(
            [word for word in str(text).split() if word not in self.stopwords_list]
        )

    def _stem_(self, text):
        return text

    def transform(self, text, debug=False):
        if self.lower:
            text = self._lower_(text)
            if debug:
                print("Method: lower")
                print("----------------------------------------------------")
                print(text)
                print("----------------------------------------------------")
        if self.remove_url:
            text = self._remove_url_(text)
            if debug:
                print("Method: remove_url")
                print("----------------------------------------------------")
                print(text)
                print("----------------------------------------------------")
        if self.remove_hashtags:
            text = self._remove_hashtags_(text)
            if debug:
                print("Method: remove_hashtags")
                print("----------------------------------------------------")
                print(text)
                print("----------------------------------------------------")
        if self.convert_emojis:
            text = self._convert_emojis_(text)
            if debug:
                print("Method: convert_emojis")
                print("----------------------------------------------------")
                print(text)
                print("----------------------------------------------------")
        if self.convert_emoticons:
            text = self._convert_emoticons_(text)
            if debug:
                print("Method: convert_emoticons")
                print("----------------------------------------------------")
                print(text)
                print("----------------------------------------------------")
        if self.normalize_inclusive_language:
            text = self._normalize_inclusive_language_(text)
            if debug:
                print("Method: normalize_inclusive_language")
                print("----------------------------------------------------")
                print(text)
                print("----------------------------------------------------")
        if self.reduce_spam:
            text = self._reduce_spam_(text)
            if debug:
                print("Method: reduce_spam")
                print("----------------------------------------------------")
                print(text)
                print("----------------------------------------------------")
        if self.remove_vowels_accents:
            text = self._remove_vowels_accents_(text)
            if debug:
                print("Method: remove_vowels_accents")
                print("----------------------------------------------------")
                print(text)
                print("----------------------------------------------------")
        if self.remove_punctuation:
            text = self._remove_punctuation_(text)
            if debug:
                print("Method: remove_punctuation")
                print("----------------------------------------------------")
                print(text)
                print("----------------------------------------------------")
        if self.remove_unprintable:
            text = self._remove_unprintable_(text)
            if debug:
                print("Method: remove_unprintable")
                print("----------------------------------------------------")
                print(text)
                print("----------------------------------------------------")
        if self.remove_stopwords:
            text = self._remove_stopwords_(text)
            if debug:
                print("Method: remove_stopwords")
                print("----------------------------------------------------")
                print(text)
                print("----------------------------------------------------")
        if self.stem:
            text = self._stem_(text)
            if debug:
                print("Method: stem")
                print("----------------------------------------------------")
                print(text)
                print("----------------------------------------------------")
        if self.preserve_emoticons:
            text = self._preserve_emoticons_(text)
            if debug:
                print("Method: preserve_emoticons")
                print("----------------------------------------------------")
                print(text)
                print("----------------------------------------------------")
        if self.preserve_emojis:
            text = self._preserve_emojis_(text)
            if debug:
                print("Method: preserve_emojis")
                print("----------------------------------------------------")
                print(text)
                print("----------------------------------------------------")
        if self.remove_numbers:
            text = self._remove_numbers_(text)
            if debug:
                print("Method: remove_numbers")
                print("----------------------------------------------------")
                print(text)
                print("----------------------------------------------------")
        return text.strip().replace("  ", " ")


if __name__ == "__main__":
    # Test SpanishPreprocess
    sp = SpanishPreprocess(
        lower=True,
        remove_url=True,
        remove_hashtags=True,
        preserve_emoticons=True,
        preserve_emojis=True,
        convert_emoticons=False,
        convert_emojis=False,
        normalize_inclusive_language=True,
        reduce_spam=True,
        remove_vowels_accents=True,
        remove_punctuation=True,
        remove_unprintable=True,
        remove_numbers=True,
        remove_stopwords=False,
        stopwords_list=None,
        stem=False,
    )

    test_text = """𝓣𝓮𝔁𝓽𝓸 𝓭𝓮 𝓹𝓻𝓾𝓮𝓫𝓪

Este es un texto de prueba :) a continuación voy a insertar una noticia de prueba 🤭👀😅

https://www.biobiochile.cl/noticias/nacional/chile/2022/07/27/presidente-boric-anuncia-copago-cero-para-fonasa-con-guinos-al-apruebo.shtml 

Me gusta la LINGÜÍSTICA y los ñandúes… También los pingüinos 🐧🐧🐧. 

Tengo una deuda de $10.000.000.000, pero tengo 2000 vacas. Mi teléfono es +5698791045"""
    print(sp.transform(test_text, debug=True))
