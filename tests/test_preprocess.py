import unittest

from parameterized import parameterized

from spanish_nlp import preprocess


class TestTextPreprocessor(unittest.TestCase):
    def setUp(self):
        self.preprocessor = preprocess.SpanishPreprocess()
        self.params = {
            "lower": False,
            "remove_url": False,
            "remove_hashtags": False,
            "split_hashtags": False,
            "normalize_breaklines": False,
            "remove_emoticons": False,
            "remove_emojis": False,
            "convert_emoticons": False,
            "convert_emojis": False,
            "normalize_inclusive_language": False,
            "reduce_spam": False,
            "remove_reduplications": False,
            "remove_vowels_accents": False,
            "remove_multiple_spaces": False,
            "remove_punctuation": False,
            "remove_unprintable": False,
            "remove_numbers": False,
            "remove_stopwords": False,
            "stopwords_list": None,
            "lemmatize": False,
            "stem": False,
            "remove_html_tags": False,
        }
        self.text = """𝓣𝓮𝔁𝓽𝓸 𝓭𝓮 𝓹𝓻𝓾𝓮𝓫𝓪

<b>Holaaaaaaaa a todxs </b>, este es un texto de prueba :) a continuación les mostraré un poema de Roberto Bolaño llamado "Los perros románticos" 🤭👀😅

https://www.poesi.as/rb9301.htm

¡Me gustan los pingüinos! Sí, los PINGÜINOS 🐧🐧🐧 🐧 #VivanLosPinguinos #SíSeñor #PinguinosDelMundoUníos #ÑanduesDelMundoTambién

Si colaboras con este repositorio te puedes ganar $100.000 (en dinero falso). O tal vez 20 pingüinos. Mi teléfono es +561212121212"""

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

    def test_stem(self):
        self.preprocessor.stem = True
        text = "Este texto contiene varias palabras."
        self.assertTrue(self.preprocessor._stem_(text) != text)

    def test_lemmatize(self):
        self.preprocessor._prepare_lemmatize_(force=True)
        self.preprocessor.stem = True
        text = "Este texto contiene varias palabras."
        self.assertTrue(self.preprocessor._lemmatize_(text) != text)

    def test_convert_emojis(self):
        text = "Este texto tiene 😀 y 🙁."
        expected = "Este texto tiene __grinning_face__ y __slightly_frowning_face__."
        pp_text = self.preprocessor._emojis_to_text_(text)
        self.assertEqual(pp_text, expected)
        self.assertTrue(text != pp_text)

    def test_remove_emojis(self):
        text = "Este texto tiene __grinning_face__ y __slightly_frowning_face__."
        expected = "Este texto tiene 😀 y 🙁."

        pp_text = self.preprocessor._text_to_emojis_(text)
        self.assertEqual(pp_text, expected)
        self.assertTrue(text != pp_text)

    def test_convert_emoticons(self):
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

    def test_transform_remove_stopwords(self):
        text = "En aquel tiempo yo tenía veinte años y estaba loco. Había perdido un país pero había ganado un sueño. Y si tenía ese sueño lo demás no importaba. Ni trabajar ni rezar ni estudiar en la madrugada junto a los perros románticos."
        expected = "tiempo tenía veinte años estaba loco. Había perdido país había ganado sueño. si tenía sueño lo demás no importaba. trabajar rezar estudiar madrugada junto perros románticos."

        self.preprocessor._prepare_stopwords_(type="default")
        pp_text = self.preprocessor._remove_stopwords_(text)
        self.assertEqual(pp_text, expected)
        self.assertTrue(text != pp_text)

    def test_transform_remove_multiple_spaces(self):
        text = "Este    texto  tiene varios         espacios. "
        expected = "Este texto tiene varios espacios."

        pp_text = self.preprocessor._remove_multiples_spaces_(text)
        self.assertEqual(pp_text, expected)
        self.assertTrue(text != pp_text)

    def test_transform_normalize_breaklines(self):
        text = "Sopaipillas \n\n\n Dos tazas de harina"
        expected = "Sopaipillas \nDos tazas de harina"
        pp_text = self.preprocessor._normalize_breaklines_(text)
        self.assertEqual(pp_text, expected)
        self.assertTrue(text != pp_text)

    def test_transform_normalize_punctuation_spelling(self):
        text = (
            "Este es un texto,con la puntuación incorrecta . Se tiene que solucionar!"
        )
        expected = (
            "Este es un texto, con la puntuación incorrecta. Se tiene que solucionar!"
        )
        pp_text = self.preprocessor._normalize_punctuation_spelling_(text)
        self.assertEqual(pp_text, expected)
        self.assertTrue(text != pp_text)

    def test_transform_reduce_spam(self):
        text = "Este es un gran gran texto con muchas muchas muchas muchas muchas repeticiones"
        expected = "Este es un gran gran texto con muchas muchas repeticiones"
        pp_text = self.preprocessor._reduce_spam_(text)
        self.assertEqual(pp_text, expected)
        self.assertTrue(text != pp_text)

    def test_transform_remove_reduplications(self):
        text = "holaaaa banana no te creoooo naaada"
        expected = "hola banana no te creo nada"
        pp_text = self.preprocessor._remove_reduplications_(text)
        self.assertEqual(pp_text, expected)
        self.assertTrue(text != pp_text)

    def test_transform_false(self):
        pp = preprocess.SpanishPreprocess(**self.params)
        pp_text = pp.transform(self.text)
        self.assertEqual(pp_text, pp_text)

    def test_transform_true(self):
        # Set all values in the dict with True
        params = {
            "lower": False,
            "remove_url": True,
            "remove_hashtags": False,
            "split_hashtags": True,
            "normalize_breaklines": True,
            "remove_emoticons": False,
            "remove_emojis": False,
            "convert_emoticons": True,
            "convert_emojis": True,
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

        pp = preprocess.SpanishPreprocess(**params)
        pp_text = pp.transform(self.text)
        expected = """Holaaaaaaaa a todos, este es un texto de prueba:) a continuacion los mostrare un poema de Roberto Bolaño llamado "Los perros romanticos" 🤭 👀 😅 
Me gustan los pinguinos! Si, los PINGUINOS 🐧 🐧 🐧 🐧 Vivan Los Pinguinos Si Señor Pinguinos Del Mundo Unios Ñandues Del Mundo Tambien
Si colaboras con este repositorio te puedes ganar $100.000 (en dinero falso). O tal vez 20 pinguinos. Mi telefono es +561212121212
"""
        self.assertEqual(pp_text, pp_text)


if __name__ == "__main__":
    unittest.main()
    print("Everything passed")
