import unittest
from spanish_nlp import preprocess


class TestTextPreprocessor(unittest.TestCase):
    def setUp(self):
        self.preprocessor = preprocess.SpanishPreprocess()

    def test_lower(self):
        text = "Ejemplo de TEXTO con mayúsculas."
        expected = "ejemplo de texto con mayúsculas."
        self.assertEqual(self.preprocessor._lower_(text), expected)

    def test_split_hashtags1(self):
        text = "Esto es un #ejemplo de texto con #hashtags"
        expected = "Esto es un ejemplo de texto con hashtags"
        self.assertEqual(self.preprocessor._split_hashtags_(text), expected)

    def test_split_hashtags2(self):
        text = "esto es #unEjemplo de texto con #hashtags"
        expected = "esto es un Ejemplo de texto con hashtags"
        self.assertEqual(self.preprocessor._split_hashtags_(text), expected)

    def test_split_hashtags3(self):
        text = "esto es #UnEjemplo de texto con #hashtags"
        expected = "esto es Un Ejemplo de texto con hashtags"
        self.assertEqual(self.preprocessor._split_hashtags_(text), expected)

    def test_split_hashtags4(self):
        text = "esto es un #hashtag, pero 4gcf#assf y 13#3 no lo son"
        expected = "esto es un hashtag, pero 4gcf#assf y 13#3 no lo son"
        self.assertEqual(self.preprocessor._split_hashtags_(text), expected)

    def test_remove_url1(self):
        self.preprocessor.remove_url = True
        text = "Este texto contiene una URL: https://www.ejemplo.com"
        expected = "Este texto contiene una URL: "
        self.assertEqual(self.preprocessor._remove_url_(text), expected)

    def test_remove_url2(self):
        self.preprocessor.remove_url = True
        text = "Este texto contiene una URL https://www.ejemplo.com/hola/test?param=1&param2=2 con parámetros."
        expected = "Este texto contiene una URL con parámetros."
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

    # TODO: Fix the following tests

    # def test_convert_emojis(self):
    #     self.preprocessor.convert_emojis = True
    #     text = "Este texto tiene 😀 y 🙁."
    #     expected = "Este texto tiene y ."
    #     self.assertEqual(self.preprocessor._

    # def test_remove_emojis(self):
    #     self.preprocessor.remove_emojis = True
    #     text = "Este texto tiene 😀 y 🙁."
    #     expected = "Este texto tiene  y ."
    #     self.assertEqual(self.preprocessor.transform(text), expected)

    # def test_convert_emoticons(self):
    #     self.preprocessor.convert_emoticons = True
    #     text = "Este texto tiene :), :(, y :D."
    #     expected = "Este texto tiene cara feliz, cara triste, y cara con sonrisa grande."
    #     self.assertEqual(self.preprocessor.transform(text), expected)

    # def test_remove_emoticons(self):
    #     self.preprocessor.remove_emoticons = True
    #     text = "Este texto tiene :), :(, y :D."
    #     expected = "Este texto tiene , , y ."
    #     self.assertEqual(self.preprocessor.transform(text), expected)

    # def test_normalize_inclusive_language(self):
    #     self.preprocessor.normalize_inclusive_language = True
    #     text = "hola a todxs"
    #     expected = "hola a todoxs"
    #     self.assertEqual(self.preprocessor.transform(text), expected)

    # def test_transform_remove_stopwords(self):
    #     t = TextPreprocessor(remove_stopwords=True, language="spanish")
    #     result = t.transform("Este es un texto de prueba con muchas palabras comunes en español")
    #     self.assertEqual(result, "texto prueba palabras comunes español")

    # def test_transform_remove_multiple_spaces(self):
    #     t = TextPreprocessor(remove_multiple_spaces=True)
    #     result = t.transform("Este es     un texto con espacios   multiples ")
    #     self.assertEqual(result, "Este es un texto con espacios multiples")

    # def test_transform_normalize_breaklines(self):
    #     t = TextPreprocessor(normalize_breaklines=True)
    #     result = t.transform("Este es\nun texto\ncon saltos de\nlínea")
    #     self.assertEqual(result, "Este es un texto con saltos de línea")

    # def test_transform_normalize_punctuation_spelling(self):
    #     t = TextPreprocessor(normalize_punctuation_spelling=True)
    #     result = t.transform("Este es un texto con: varias, puntuaciones.. y algunos acentos")
    #     self.assertEqual(result, "Este es un texto con varias puntuaciones y algunos acentos")

    # def test_transform_reduce_spam(self):
    #     t = TextPreprocessor(reduce_spam=True)
    #     result = t.transform("Oferta especial!!! Compra ya nuestro producto!!!")
    #     self.assertEqual(result, "Oferta especial Compra ya nuestro producto")

    # def test_transform_remove_reduplications(self):
    #     t = TextPreprocessor(remove_reduplications=True)
    #     result = t.transform("Este es un texto con con algunas algunas palabras repetidas")
    #     self.assertEqual(result, "Este es un texto con algunas palabras repetidas")

    # def test_transform_debug(self):
    #     t = TextPreprocessor(remove_numbers=True, lower=True, debug=True)
    #     result = t.transform("Este es un TEXTO con 123 números")
    #     self.assertEqual(result, "Este es un texto con números")


if __name__ == "__main__":
    unittest.main()
    print("Everything passed")
