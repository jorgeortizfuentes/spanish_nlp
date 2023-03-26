import unittest

from spanish_nlp import augmentation

class TestSpelling(unittest.TestCase):
    def setUp(self):
        self.keyboard_augmentator = augmentation.Spelling(method="keyboard")
        self.ocr_augmentator = augmentation.Spelling(method="ocr")
        self.random_augmentator = augmentation.Spelling(method="random")
        self.orthography_augmentator = augmentation.Spelling(
            method="orthography"
        )
        self.remove_punctuation_augmentator = augmentation.Spelling(
            method="remove_punctuation"
        )
        self.remove_accents_augmentator = augmentation.Spelling(
            method="remove_accents"
        )
        self.all_augmentator = augmentation.Spelling(method="all")
        self.text = "En aquel tiempo yo tenía veinte años y estaba loco. Había perdido un país pero había ganado un sueño. Y si tenía ese sueño lo demás no importaba. Ni trabajar ni rezar ni estudiar en la madrugada junto a los perros románticos."
        
    def print_augmentations(self, original, augmentations, method):
        # Save in a file "augmentations_test.txt"
        with open("tests/augmentations_test.txt", "a") as f:
            f.write(f"*** SPELLING DATA AUGMENTATION: {method} ***" + "\n")
            f.write(f"Original: {original}" + "\n")
            for i in range(len(augmentations)):
                f.write(f"Augmentation {i+1}: {augmentations[i]}" + "\n")
            f.write("-----------------------------" + "\n")

    def test_keyboard_augment_(self):
        text_aug = self.keyboard_augmentator.augment(self.text, 1)
        self.print_augmentations(self.text, text_aug, method="keyboard")
        for i in range(len(text_aug)):
            self.assertFalse(self.text == text_aug[i])
            self.assertFalse(self.text == "")

    def test_ocr_augment_(self):
        text_aug = self.ocr_augmentator.augment(self.text, 1)
        self.print_augmentations(self.text, text_aug, method="ocr")
        for i in range(len(text_aug)):
            self.assertFalse(self.text == text_aug[i])
            self.assertFalse(self.text == "")

    def test_random_augment_(self):
        text_aug = self.random_augmentator.augment(self.text, 1)
        self.print_augmentations(self.text, text_aug, method="random")
        for i in range(len(text_aug)):
            self.assertFalse(self.text == text_aug[i])
            self.assertFalse(self.text == "")

    def test_orthography_augment(self):
        text_aug = self.orthography_augmentator.augment(self.text, 1)
        self.print_augmentations(self.text, text_aug, method="orthography")
        for i in range(len(text_aug)):
            self.assertFalse(self.text == text_aug[i])
            self.assertFalse(self.text == "")

    def test_remove_punctuation_augment(self):
        text_aug = self.remove_punctuation_augmentator.augment(self.text, 1)
        self.print_augmentations(self.text, text_aug, method="orthography")
        for i in range(len(text_aug)):
            self.assertFalse(self.text == text_aug[i])
            self.assertFalse(self.text == "")

    def test_remove_accents_augment(self):
        text_aug = self.remove_accents_augmentator.augment(self.text, 1)
        self.print_augmentations(self.text, text_aug, method="orthography")
        for i in range(len(text_aug)):
            self.assertFalse(self.text == text_aug[i])
            self.assertFalse(self.text == "")

    def test_all_augment(self):
        text_aug = self.all_augmentator.augment(self.text, 1)
        self.print_augmentations(self.text, text_aug, method="all")
        for i in range(len(text_aug)):
            self.assertFalse(self.text == text_aug[i])
            self.assertFalse(self.text == "")


if __name__ == "__main__":
    unittest.main()
