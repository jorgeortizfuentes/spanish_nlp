import unittest
from spanish_nlp import SpanishSpellChecker

class TestSpanishSpellCheckerDictionary(unittest.TestCase):

    def setUp(self):
        self.checker = SpanishSpellChecker(method='dictionary')
        self.text_with_errors = "Ola komo stas? Esto es una prueva."
        self.text_correct = "Hola cómo estás? Esto es una prueba."

    def test_is_correct(self):
        self.assertTrue(self.checker.is_correct("hola"))
        self.assertTrue(self.checker.is_correct("cómo"))
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
        self.assertIn("Ola", errors) # Case might matter depending on implementation detail, pyspellchecker is case-insensitive internally
        self.assertIn("komo", errors)
        self.assertIn("stas", errors)
        self.assertIn("prueva", errors)
        self.assertNotIn("Esto", errors)
        self.assertNotIn("es", errors)
        self.assertNotIn("una", errors)

        errors_correct = self.checker.find_errors(self.text_correct)
        self.assertEqual(len(errors_correct), 0)

    def test_correct_text(self):
        corrected_text = self.checker.correct_text(self.text_with_errors)
        # Allow for variations in capitalization if the base method handles it
        self.assertEqual(corrected_text.lower(), self.text_correct.lower())

        corrected_correct_text = self.checker.correct_text(self.text_correct)
        self.assertEqual(corrected_correct_text.lower(), self.text_correct.lower())

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
