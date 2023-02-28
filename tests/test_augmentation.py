import unittest

from spanish_nlp import augmentation


class TestSpelling(unittest.TestCase):
    def setUp(self):
        self.keyboard_augmentator = augmentation.Spelling(method="keyboard")
        self.ocr_augmentator = augmentation.Spelling(method="ocr")
        self.random_augmentator = augmentation.Spelling(method="random")
        self.orthography_augmentator = augmentation.Spelling(method="orthography")
        self.all_augmentator = augmentation.Spelling(method="all")

    def print_augmentations(self, original, augmentations, method):
        # Save in a file "augmentations_test.txt"
        with open("tests/augmentations_test.txt", "a") as f:
            f.write(f"*** SPELLING DATA AUGMENTATION: {method} ***" + "\n")
            f.write(f"Original: {original}" + "\n")
            for i in range(len(augmentations)):
                f.write(f"Augmentation {i+1}: {augmentations[i]}" + "\n")
            f.write("-----------------------------" + "\n")

    def test_keyboard_augment_(self):
        text = "En aquel tiempo yo tenía veinte años y estaba loco. Había perdido un país pero había ganado un sueño. Y si tenía ese sueño lo demás no importaba. Ni trabajar ni rezar ni estudiar en la madrugada junto a los perros románticos."
        text_aug = self.keyboard_augmentator.augment(text, 1)
        self.print_augmentations(text, text_aug, method="keyboard")
        for i in range(len(text_aug)):
            self.assertFalse(text == text_aug[i])
            self.assertFalse(text == "")

    def test_ocr_augment_(self):
        text = "En aquel tiempo yo tenía veinte años y estaba loco. Había perdido un país pero había ganado un sueño. Y si tenía ese sueño lo demás no importaba. Ni trabajar ni rezar ni estudiar en la madrugada junto a los perros románticos."
        text_aug = self.ocr_augmentator.augment(text, 1)
        self.print_augmentations(text, text_aug, method="ocr")
        for i in range(len(text_aug)):
            self.assertFalse(text == text_aug[i])
            self.assertFalse(text == "")

    def test_random_augment_(self):
        text = "En aquel tiempo yo tenía veinte años y estaba loco. Había perdido un país pero había ganado un sueño. Y si tenía ese sueño lo demás no importaba. Ni trabajar ni rezar ni estudiar en la madrugada junto a los perros románticos."
        text_aug = self.random_augmentator.augment(text, 1)
        self.print_augmentations(text, text_aug, method="random")
        for i in range(len(text_aug)):
            self.assertFalse(text == text_aug[i])
            self.assertFalse(text == "")

    def test_orthography_augment(self):
        text = "En aquel tiempo yo tenía veinte años y estaba loco. Había perdido un país pero había ganado un sueño. Y si tenía ese sueño lo demás no importaba. Ni trabajar ni rezar ni estudiar en la madrugada junto a los perros románticos."
        text_aug = self.orthography_augmentator.augment(text, 1)
        self.print_augmentations(text, text_aug, method="orthography")
        for i in range(len(text_aug)):
            self.assertFalse(text == text_aug[i])
            self.assertFalse(text == "")

    def test_all_augment(self):
        text = "En aquel tiempo yo tenía veinte años y estaba loco. Había perdido un país pero había ganado un sueño. Y si tenía ese sueño lo demás no importaba. Ni trabajar ni rezar ni estudiar en la madrugada junto a los perros románticos."
        text_aug = self.all_augmentator.augment(text, 1)
        self.print_augmentations(text, text_aug, method="all")
        for i in range(len(text_aug)):
            self.assertFalse(text == text_aug[i])
            self.assertFalse(text == "")


class TestMasked(unittest.TestCase):
    def setUp(self):
        self.sustitute_augmentor = augmentation.Masked(method="sustitute", model="dccuchile/bert-base-spanish-wwm-cased", aug_percent=0.5)
        self.insert_augmentor = augmentation.Masked(method="insert", model="dccuchile/bert-base-spanish-wwm-cased", aug_percent=0.5)

    def print_augmentations(self, original, augmentations, method):
        # Save in a file "augmentations_test.txt"
        with open("tests/augmentations_test.txt", "a") as f:
            f.write(f"*** Masked DATA AUGMENTATION: {method} ***" + "\n")
            f.write(f"Original: {original}" + "\n")
            for i in range(len(augmentations)):
                f.write(f"Augmentation {i+1}: {augmentations[i]}" + "\n")
            f.write("-----------------------------" + "\n")

    def test_sustitute(self):
        text = "En aquel tiempo yo tenía veinte años y estaba loco. Había perdido un país pero había ganado un sueño. Y si tenía ese sueño lo demás no importaba. Ni trabajar ni rezar ni estudiar en la madrugada junto a los perros románticos."
        text_aug = self.sustitute_augmentor.augment(text, 1)
        self.print_augmentations(text, text_aug, method="substitute")
        for i in range(len(text_aug)):
            self.assertFalse(text == text_aug[i])
            self.assertFalse(text == "")

    def test_insert(self):
        text = "En aquel tiempo yo tenía veinte años y estaba loco. Había perdido un país pero había ganado un sueño. Y si tenía ese sueño lo demás no importaba. Ni trabajar ni rezar ni estudiar en la madrugada junto a los perros románticos."
        text_aug = self.insert_augmentor.augment(text, 1)
        self.print_augmentations(text, text_aug, method="insert")
        for i in range(len(text_aug)):
            self.assertFalse(text == text_aug[i])
            self.assertFalse(text == "")

    def test_sustitute_large(self):
        text = 10*"En aquel tiempo yo tenía veinte años y estaba loco. Había perdido un país pero había ganado un sueño. Y si tenía ese sueño lo demás no importaba. Ni trabajar ni rezar ni estudiar en la madrugada junto a los perros románticos."
        text_aug = self.sustitute_augmentor.augment(text, 1)
        self.print_augmentations(text, text_aug, method="substitute")
        for i in range(len(text_aug)):
            self.assertTrue(text == text_aug[i])
            self.assertFalse(text == "")

    def test_insert_large(self):
        text = 10*"En aquel tiempo yo tenía veinte años y estaba loco. Había perdido un país pero había ganado un sueño. Y si tenía ese sueño lo demás no importaba. Ni trabajar ni rezar ni estudiar en la madrugada junto a los perros románticos."
        text_aug = self.insert_augmentor.augment(text, 1)
        self.print_augmentations(text, text_aug, method="insert")
        for i in range(len(text_aug)):
            self.assertTrue(text == text_aug[i])
            self.assertFalse(text == "")


if __name__ == "__main__":
    unittest.main(buffer=False, defaultTest="TestMasked")
    unittest.main(buffer=False, defaultTest='TestSpelling')
