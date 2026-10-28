"""
Test suite for SpanishPreprocess. Designed for "barebones" (no extras)
and full spanish_nlp[all] installations.

How it works:
- OPTIONAL_DEPENDENCY_METHODS maps optional features to required packages.
- Tests for features requiring optional modules validate ModuleNotFoundError
  with personalized messages when dependencies are missing, while core features
  are always tested.
"""

import re
import unittest

import pytest
from parameterized import parameterized

from spanish_nlp import SpanishPreprocess

# ---------------------------------------------------------------------------
# Optional dependency mapping per method/feature.
# Key: method or feature name in SpanishPreprocess.
# Value: required pip package or module name.
# ---------------------------------------------------------------------------
OPTIONAL_DEPENDENCY_METHODS = {
    "_stem_": "nltk",
    "_lemmatize_": "es_core_news_sm",
    "_prepare_lemmatize_": "es_core_news_sm",
    "_prepare_stopwords_[nltk]": "nltk",
    "_prepare_stopwords_[spacy]": "es_core_news_sm",
    "_emojis_to_text_": "emoji",
    "_text_to_emojis_": "emoji",
}

# Core methods that must work without optional dependencies
CORE_METHODS = [
    "_lower_",
    "_remove_url_",
    "_remove_hashtags_",
    "_split_hashtags_",
    "_normalize_breaklines_",
    "_emoticons_to_text_",
    "_text_to_emoticons_",
    "_normalize_inclusive_language_",
    "_reduce_spam_",
    "_remove_reduplications_",
    "_remove_vowels_accents_",
    "_remove_punctuation_",
    "_remove_unprintable_",
    "_remove_numbers_",
    "_remove_stopwords_[default/extended/list]",
    "_remove_multiples_spaces_",
    "_normalize_punctuation_spelling_",
    "_remove_html_tags_",
]


MSG_ERROR_TEMPLATE = (
    "'{module}' not found.\n"
    "Please install with either: \n"
    "- pip install 'spanish_nlp[{extra}]'\n"
    "- uv add 'spanish_nlp[{extra}]'"
)


# ---------------------------------------------------------------------------
# Core tests: Must ALWAYS pass without any optional dependencies installed.
# ---------------------------------------------------------------------------
class TestSpanishPreprocessCore(unittest.TestCase):
    """Tests functions that do not rely on external optional libraries."""

    def setUp(self):
        # Setting default parameters avoids instantiating nltk/spacy lists by default.
        self.preprocessor = SpanishPreprocess()

    def test_lower(self):
        text = "Ejemplo de TEXTO con mayúsculas."
        expected = "ejemplo de texto con mayúsculas."
        self.assertEqual(self.preprocessor._lower_(text), expected)

    @parameterized.expand(
        [
            (
                "Esto es un #ejemplo de texto con #hashtags",
                "Esto es un ejemplo de texto con hashtags",
            ),
            (
                "esto es #unEjemplo de texto con #hashtags",
                "esto es un Ejemplo de texto con hashtags",
            ),
            (
                "esto es #UnEjemplo de texto con #hashtags",
                "esto es Un Ejemplo de texto con hashtags",
            ),
            (
                "esto es un #hashtag, pero 4gcf#assf y 13#3 no lo son",
                "esto es un hashtag, pero 4gcf#assf y 13#3 no lo son",
            ),
        ]
    )
    def test_split_hashtags(self, text, expected):
        self.assertEqual(self.preprocessor._split_hashtags_(text), expected)

    @parameterized.expand(
        [
            (
                "Este texto contiene una URL: https://www.ejemplo.com",
                "Este texto contiene una URL: ",
            ),
            (
                "Este texto contiene una URL https://www.ejemplo.com/hola/test?param=1&param2=2 con parámetros.",
                "Este texto contiene una URL con parámetros.",
            ),
        ]
    )
    def test_remove_url(self, text, expected):
        self.preprocessor.remove_url = True
        self.assertEqual(self.preprocessor._remove_url_(text), expected)

    def test_remove_html_tags(self):
        self.preprocessor.remove_html_tags = True
        text = "<p>Este texto</p> <b>contiene</b> <i>etiquetas HTML</i>."
        expected = "Este texto contiene etiquetas HTML."
        self.assertEqual(self.preprocessor._remove_html_tags_(text), expected)

    def test_remove_numbers(self):
        self.preprocessor.remove_numbers = True
        text = "Este texto tiene números como 123 y 45678."
        expected = "Este texto tiene números como  y ."
        self.assertEqual(self.preprocessor._remove_numbers_(text), expected)

    def test_remove_hashtags(self):
        self.preprocessor.remove_hashtags = True
        text = "Este texto tiene #hashtags y #mencionados."
        expected = "Este texto tiene y ."
        self.assertEqual(self.preprocessor._remove_hashtags_(text), expected)

    def test_convert_emoticons(self):
        # Text emoticons like :) and :( are handled via spanish_nlp's internal dictionary.
        text = "Este texto tiene :) y :(."
        expected = "Este texto tiene __happy_face_or_smiley_2__ y __frown_sad_andry_or_pouting_3__."
        pp_text = self.preprocessor._emoticons_to_text_(text)
        self.assertEqual(pp_text, expected)
        self.assertTrue(text != pp_text)

    def test_remove_emoticons(self):
        text = "Este texto tiene __happy_face_or_smiley_2__ y __frown_sad_andry_or_pouting_3__."
        expected = "Este texto tiene :) y :(."
        pp_text = self.preprocessor._text_to_emoticons_(text)
        self.assertEqual(pp_text, expected)
        self.assertTrue(text != pp_text)

    def test_normalize_inclusive_language(self):
        text = "hola a todxs un saludo a mis amiges"
        expected = "hola a todos un saludo a mis amigos"
        pp_text = self.preprocessor._normalize_inclusive_language_(text)
        self.assertEqual(pp_text, expected)
        self.assertTrue(text != pp_text)

    def test_remove_stopwords_default(self):
        text = (
            "En aquel tiempo yo tenía veinte años y estaba loco. Había "
            "perdido un país pero había ganado un sueño. Y si tenía ese "
            "sueño lo demás no importaba. Ni trabajar ni rezar ni estudiar "
            "en la madrugada junto a los perros románticos."
        )
        expected = (
            "tiempo tenía veinte años estaba loco. Había perdido país "
            "había ganado sueño. si tenía sueño lo demás no importaba. "
            "trabajar rezar estudiar madrugada junto perros románticos."
        )
        self.preprocessor._prepare_stopwords_(type="default")
        pp_text = self.preprocessor._remove_stopwords_(text)
        self.assertEqual(pp_text, expected)
        self.assertTrue(text != pp_text)

    def test_remove_stopwords_custom_list(self):
        text = "el perro come pan y agua"
        self.preprocessor._prepare_stopwords_(type=["el", "y"])
        pp_text = self.preprocessor._remove_stopwords_(text)
        self.assertEqual(pp_text, "perro come pan agua")

    def test_remove_multiple_spaces(self):
        text = "Este    texto  tiene varios         espacios. "
        expected = "Este texto tiene varios espacios."
        pp_text = self.preprocessor._remove_multiples_spaces_(text)
        self.assertEqual(pp_text, expected)
        self.assertTrue(text != pp_text)

    def test_normalize_breaklines(self):
        text = "Sopaipillas \n\n\n Dos tazas de harina"
        expected = "Sopaipillas \nDos tazas de harina"
        pp_text = self.preprocessor._normalize_breaklines_(text)
        self.assertEqual(pp_text, expected)
        self.assertTrue(text != pp_text)

    def test_normalize_punctuation_spelling(self):
        text = (
            "Este es un texto,con la puntuación incorrecta . Se tiene que solucionar!"
        )
        expected = (
            "Este es un texto, con la puntuación incorrecta. Se tiene que solucionar!"
        )
        pp_text = self.preprocessor._normalize_punctuation_spelling_(text)
        self.assertEqual(pp_text, expected)
        self.assertTrue(text != pp_text)

    def test_reduce_spam(self):
        text = "Este es un gran gran texto con muchas muchas muchas muchas muchas repeticiones"
        expected = "Este es un gran gran texto con muchas muchas repeticiones"
        pp_text = self.preprocessor._reduce_spam_(text)
        self.assertEqual(pp_text, expected)
        self.assertTrue(text != pp_text)

    def test_remove_reduplications(self):
        text = "holaaaa banana no te creoooo naaada"
        expected = "hola banana no te creo nada"
        pp_text = self.preprocessor._remove_reduplications_(text)
        self.assertEqual(pp_text, expected)
        self.assertTrue(text != pp_text)

    def test_remove_vowels_accents(self):
        text = "áéíóú ÁÉÍÓÚ ñÑ"
        expected = "aeiou AEIOU ñÑ"
        self.assertEqual(self.preprocessor._remove_vowels_accents_(text), expected)

    def test_remove_punctuation(self):
        text = "¡Hola! ¿Cómo estás?, muy bien."
        pp_text = self.preprocessor._remove_punctuation_(text)
        self.assertNotIn("!", pp_text)
        self.assertNotIn("¿", pp_text)
        self.assertNotIn(",", pp_text)

    def test_remove_unprintable(self):
        text = "Texto normal 𝓣𝓮𝔁𝓽𝓸 con ñ y á"
        pp_text = self.preprocessor._remove_unprintable_(text)
        self.assertNotIn("𝓣", pp_text)
        self.assertIn("ñ", pp_text)
        self.assertIn("á", pp_text)

    def test_full_transform_without_optional_features(self):
        """Full transform() with external dependency features disabled;

        must work seamlessly in barebones installations.
        """
        params = {
            "lower": True,
            "remove_url": True,
            "remove_hashtags": False,
            "split_hashtags": True,
            "normalize_breaklines": True,
            "remove_emoticons": True,
            "remove_emojis": True,
            "convert_emoticons": False,
            "convert_emojis": False,
            "normalize_inclusive_language": True,
            "reduce_spam": True,
            "remove_reduplications": True,
            "remove_vowels_accents": True,
            "remove_multiple_spaces": True,
            "remove_punctuation": False,
            "remove_unprintable": True,
            "remove_numbers": False,
            "remove_stopwords": False,
            "stopwords_list": None,
            "lemmatize": False,
            "stem": False,
            "remove_html_tags": True,
        }
        pp = SpanishPreprocess(**params)
        text = (
            "<b>Holaaaaaaaa a todxs</b>, este es un texto de prueba :) "
            "https://www.ejemplo.com con #unHashtag y $100.000"
        )
        result = pp.transform(text)
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)


# ---------------------------------------------------------------------------
# Optional dependency tests: Exception handling checks when modules are missing.
# ---------------------------------------------------------------------------
class TestSpanishPreprocessOptionalNLTK(unittest.TestCase):
    def setUp(self):
        self.preprocessor = SpanishPreprocess()
        self.msg_error = MSG_ERROR_TEMPLATE.format(
            module="nltk",
            extra="nltk",
        )

    def test_stem(self):
        self.preprocessor.stem = True
        text = "Este texto contiene varias palabras."

        with pytest.raises(ModuleNotFoundError, match=re.escape(self.msg_error)):
            self.preprocessor._stem_(text)

    def test_stopwords_nltk(self):
        with pytest.raises(ModuleNotFoundError, match=re.escape(self.msg_error)):
            self.preprocessor._prepare_stopwords_(type="nltk")


class TestSpanishPreprocessOptionalSpacy(unittest.TestCase):
    def setUp(self):
        self.preprocessor = SpanishPreprocess()
        self.msg_error = MSG_ERROR_TEMPLATE.format(
            module="es_core_news_sm",
            extra="spacy",
        )

    def test_lemmatize(self):
        with pytest.raises(ModuleNotFoundError, match=re.escape(self.msg_error)):
            self.preprocessor._prepare_lemmatize_(force=True)

    def test_stopwords_spacy(self):
        with pytest.raises(ModuleNotFoundError, match=re.escape(self.msg_error)):
            self.preprocessor._prepare_stopwords_(type="spacy")


class TestSpanishPreprocessOptionalEmoji(unittest.TestCase):
    def setUp(self):
        self.preprocessor = SpanishPreprocess()
        self.msg_error = MSG_ERROR_TEMPLATE.format(module="emoji", extra="emoji")

    def test_convert_emojis(self):
        text = "Este texto tiene 😀 y 🙁."
        with pytest.raises(ModuleNotFoundError, match=re.escape(self.msg_error)):
            self.preprocessor._emojis_to_text_(text)

    def test_remove_emojis(self):
        text = "Este texto tiene __grinning_face__ y __slightly_frowning_face__."
        with pytest.raises(ModuleNotFoundError, match=re.escape(self.msg_error)):
            self.preprocessor._text_to_emojis_(text)


if __name__ == "__main__":
    unittest.main()
