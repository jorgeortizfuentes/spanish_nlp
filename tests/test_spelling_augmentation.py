import pandas as pd
import pytest
from unidecode import unidecode

from spanish_nlp.augmentation import Spelling

TEXT = (
    "En aquel tiempo yo tenía veinte años y estaba loco. ¿Había perdido un país? "
    "Pero había GANADO un sueño, y si tenía ese sueño lo demás no importaba."
)
PUNCTUATION = ".,¡¿"
ACCENTS = "áéíóúÁÉÍÓÚüÜ"


def augment(method, text=TEXT, num_samples=5, **kwargs):
    return Spelling(method=method, **kwargs).augment(text, num_samples)


def count(text, chars):
    return sum(c in chars for c in text)


def test_invalid_method_raises():
    with pytest.raises(ValueError, match="Method not available"):
        Spelling(method="unknown")


def test_invalid_input_type_raises():
    with pytest.raises(ValueError, match="must be a string"):
        Spelling(method="lowercase").augment(123)


def test_list_input_returns_one_result_per_text():
    texts = ["Hola Mundo", "Otro Texto De Prueba"]
    result = Spelling(method="lowercase").augment(texts, num_samples=2)
    assert len(result) == len(texts)
    assert all(len(samples) == 2 for samples in result)


def test_series_input_returns_series():
    texts = pd.Series(["Hola Mundo", "Otro Texto De Prueba"])
    result = Spelling(method="lowercase").augment(texts, num_workers=1)
    assert isinstance(result, pd.Series)
    assert len(result) == len(texts)


@pytest.mark.parametrize("method", ["keyboard", "ocr", "random"])
def test_character_methods_replace_characters_in_place(method):
    for sample in augment(method):
        assert len(sample) == len(TEXT)
        assert sample != TEXT


def test_grapheme_spelling_changes_text():
    for sample in augment("grapheme_spelling"):
        assert sample != TEXT


def test_word_spelling_replaces_known_words():
    text = "Creo que hay gente ahí"
    for sample in augment("word_spelling", text=text, aug_percent=1.0):
        assert sample.startswith("Creo")
        assert " hay " not in sample
        assert "ahí" not in sample


def test_word_spelling_without_known_words_keeps_text():
    assert (
        augment("word_spelling", text="Nada para cambiar") == ["Nada para cambiar"] * 5
    )


@pytest.mark.parametrize(
    ("method", "chars"),
    [
        ("remove_punctuation", PUNCTUATION),
        ("remove_spaces", " "),
    ],
)
def test_remove_methods_only_remove_target_characters(method, chars):
    def strip(text):
        return "".join(c for c in text if c not in chars)

    for sample in augment(method, aug_percent=0.5):
        assert strip(sample) == strip(TEXT)
        assert 0 < count(sample, chars) < count(TEXT, chars)


def test_remove_accents_only_removes_accents():
    for sample in augment("remove_accents", aug_percent=0.5):
        assert unidecode(sample) == unidecode(TEXT)
        assert 0 < count(sample, ACCENTS) < count(TEXT, ACCENTS)


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("remove_punctuation", "".join(c for c in TEXT if c not in PUNCTUATION)),
        ("remove_spaces", TEXT.replace(" ", "")),
        ("remove_accents", "".join(unidecode(c) if c in ACCENTS else c for c in TEXT)),
    ],
)
def test_remove_methods_with_full_percent_remove_everything(method, expected):
    assert augment(method, aug_percent=1.0, num_samples=1) == [expected]


@pytest.mark.parametrize("method", ["lowercase", "uppercase", "randomcase"])
def test_case_methods_only_change_case(method):
    for sample in augment(method):
        assert sample.lower() == TEXT.lower()
        assert sample != TEXT


def test_lowercase_reduces_uppercase_letters():
    for sample in augment("lowercase"):
        assert sum(c.isupper() for c in sample) < sum(c.isupper() for c in TEXT)


def test_uppercase_increases_uppercase_letters():
    for sample in augment("uppercase"):
        assert sum(c.isupper() for c in sample) > sum(c.isupper() for c in TEXT)


def test_all_combines_augmentations_on_full_text():
    for sample in augment("all"):
        assert sample != TEXT
        assert 0.8 * len(TEXT) < len(sample) <= len(TEXT)
