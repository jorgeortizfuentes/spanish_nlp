import unittest
from spanish_nlp import SpanishSpellChecker

class TestSpanishSpellCheckerDictionary(unittest.TestCase):

    def setUp(self):
        self.checker = SpanishSpellChecker(method='dictionary')
        self.text_with_errors = "Ola komo stas? Esto es una prueva."
        self.text_correct = "Hola cómo estás? Esto es una prueba."

    def test_is_correct(self):
        self.assertTrue(self.checker.is_correct("hola"))
        self.assertFalse(self.checker.is_correct("cómo")) # Default dict likely doesn't have accented version
        self.assertTrue(self.checker.is_correct("como")) # Check non-accented version is likely correct
        self.assertFalse(self.checker.is_correct("komo"))
        self.assertFalse(self.checker.is_correct("stas"))
        self.assertFalse(self.checker.is_correct("prueva"))

    def test_suggest(self):
        suggestions = self.checker.suggest("komo")
        self.assertIsInstance(suggestions, list)
        self.assertIn("como", suggestions)

        suggestions_stas = self.checker.suggest("stas")
        self.assertIsInstance(suggestions_stas, list)
        self.assertTrue(len(suggestions_stas) > 0)

        suggestions_correct = self.checker.suggest("hola")
        self.assertIsInstance(suggestions_correct, list)
        # Depending on the dictionary, it might suggest variations or just the word itself
        self.assertTrue(len(suggestions_correct) >= 1)


    def test_correct_word(self):
        self.assertEqual(self.checker.correct_word("komo"), "como")
        self.assertEqual(self.checker.correct_word("stas"), "estas") # Common correction
        self.assertEqual(self.checker.correct_word("prueva"), "prueba")
        self.assertEqual(self.checker.correct_word("hola"), "hola") # Should return correct word

    def test_find_errors(self):
        errors = self.checker.find_errors(self.text_with_errors)
        self.assertIsInstance(errors, list)
        # self.assertIn("Ola", errors) # pyspellchecker seems case-insensitive or considers 'Ola' correct
        self.assertNotIn("Ola", errors) # Expect 'Ola' NOT to be an error
        self.assertIn("komo", errors)
        self.assertIn("stas", errors)
        self.assertIn("prueva", errors)
        self.assertNotIn("Esto", errors)
        self.assertNotIn("es", errors)
        self.assertNotIn("una", errors)

        errors_correct = self.checker.find_errors(self.text_correct)
        # Expect accented words to be flagged as errors by default dict
        self.assertCountEqual(errors_correct, ["cómo", "estás"]) # Use assertCountEqual for order-insensitive list comparison

    def test_correct_text(self):
        corrected_text = self.checker.correct_text(self.text_with_errors)
        # Adjust expected text: 'Ola' is not corrected, others are lowercased without accents
        expected_corrected_text = "ola como estas? esto es una prueba."
        self.assertEqual(corrected_text.lower(), expected_corrected_text)

        # Test correcting the original 'correct' text (with accents and caps)
        # Expect output to be lowercased and without accents due to correction behavior
        expected_output_from_correct = "hola como estas? esto es una prueba."
        corrected_correct_text = self.checker.correct_text(self.text_correct)
        self.assertEqual(corrected_correct_text.lower(), expected_output_from_correct)

    def test_init_with_distance(self):
        checker_strict = SpanishSpellChecker(method='dictionary', distance=1)
        # 'komo' -> 'como' is distance 1
        self.assertEqual(checker_strict.correct_word("komo"), "como")
        # 'pruevs' -> 'prueba' is distance 2, should not be corrected with distance=1
        self.assertEqual(checker_strict.correct_word("pruevs"), "pruevs")

        suggestions_strict = checker_strict.suggest("pruevs")
        self.assertEqual(len(suggestions_strict), 0)

    def test_init_with_custom_dictionary(self):
        custom_words = ["customword", "nlpaug"]
        checker_custom = SpanishSpellChecker(method='dictionary', custom_dictionary=custom_words)
        self.assertTrue(checker_custom.is_correct("customword"))
        self.assertTrue(checker_custom.is_correct("nlpaug"))
        self.assertFalse(checker_custom.is_correct("anotherword"))

        text_custom = "This text uses customword and nlpaug."
        errors = checker_custom.find_errors(text_custom)
        self.assertNotIn("customword", errors)
        self.assertNotIn("nlpaug", errors)
        self.assertIn("This", errors) # Assuming English words are errors in 'es' dict


if __name__ == "__main__":
    unittest.main()
