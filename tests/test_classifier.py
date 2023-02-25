import unittest

from spanish_nlp import classifiers


class TestSpanishClassifier(unittest.TestCase):
    def setUp(self):
        self.sc = classifiers.SpanishClassifier(model_name="hate_speech", device="cpu")
        self.t1 = "LAS MUJERES QUE ESTÁN EN EL CONGRESO DEBERÍAN SER EXTERMINADAS"
        self.t2 = "El presidente convocó a una reunión a los representantes de los partidos políticos"

    def test_predict(self):
        p1 = self.sc.predict(self.t1)
        # Round p1 to 0 decimals
        p1 = int(round(p1["hate_speech"], 0))
        self.assertEqual(p1, 1)
        p2 = self.sc.predict(self.t2)
        p2 = int(round(p2["not_hate_speech"], 0))
        self.assertEqual(p2, 1)


if __name__ == "__main__":
    unittest.main()
