import unittest
import pandas as pd

from spanish_nlp import augmentation

class TestMasked(unittest.TestCase):
    def setUp(self):
        self.sustitute_augmentor = augmentation.Masked(method="sustitute",
                                                       model="dccuchile/bert-base-spanish-wwm-cased",
                                                       aug_percent=0.5,
                                                       device="cpu")
        self.insert_augmentor = augmentation.Masked(method="insert",
                                                    model="dccuchile/bert-base-spanish-wwm-cased",
                                                    aug_percent=0.5,
                                                    device="cpu")

    def print_augmentations(self, original, augmentations, method):
        # Save in a file "augmentations_test.txt"
        with open("augmentations_test.txt", "a") as f:
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
        text = 30*"En aquel tiempo yo tenía veinte años y estaba loco. Había perdido un país pero había ganado un sueño. Y si tenía ese sueño lo demás no importaba. Ni trabajar ni rezar ni estudiar en la madrugada junto a los perros románticos."
        text_aug = self.sustitute_augmentor.augment(text, 1)
        self.print_augmentations(text, text_aug, method="substitute")
        for i in range(len(text_aug)):
            print(text)
            print(".....")
            print(text_aug[i])
            self.assertFalse(text == text_aug[i])
            self.assertFalse(text == "")

    def test_insert_large(self):
        text = 30*"En aquel tiempo yo tenía veinte años y estaba loco. Había perdido un país pero había ganado un sueño. Y si tenía ese sueño lo demás no importaba. Ni trabajar ni rezar ni estudiar en la madrugada junto a los perros románticos."
        text_aug = self.insert_augmentor.augment(text, 1)
        self.print_augmentations(text, text_aug, method="insert")
        for i in range(len(text_aug)):
            self.assertFalse(text == text_aug[i])
            self.assertFalse(text == "")

    def test_pandas(self):
        texts = ["soy un texto para probar pandas en ste test"]*100
        df = pd.DataFrame({"text": texts})
        df["sustitute"] = self.sustitute_augmentor.augment(df["text"], num_workers=1)
        df["insert"] = self.insert_augmentor.augment(df["text"], num_workers=1)
        # Compare if df["text"] is equal to df["sustitute"] or df["insert"]
        self.assertFalse(df["text"].equals(df["sustitute"]))
        self.assertFalse(df["text"].equals(df["insert"]))


if __name__ == "__main__":
    unittest.main()