import pytest

from spanish_nlp.augmentation import Masked

TINY_MODEL = "hf-internal-testing/tiny-random-BertForMaskedLM"
SPANISH_MODEL = "dccuchile/bert-base-spanish-wwm-cased"
METHODS = ["sustitute", "insert"]
SHORT_TEXT = "En aquel tiempo yo tenía veinte años y estaba loco."
LONG_TEXT = " ".join([SHORT_TEXT] * 15)
PARAGRAPH = (
    "En aquel tiempo yo tenía veinte años y estaba loco. Había perdido un país "
    "pero había ganado un sueño. Y si tenía ese sueño lo demás no importaba. Ni "
    "trabajar ni rezar ni estudiar en la madrugada junto a los perros románticos."
)


@pytest.fixture(scope="module", params=METHODS)
def augmenter(request):
    return Masked(method=request.param, model=TINY_MODEL, aug_percent=0.3, device="cpu")


def normalize(augmenter, text):
    tokenizer = augmenter.tokenizer
    return tokenizer.decode(tokenizer(text)["input_ids"], skip_special_tokens=True)


def test_invalid_method_raises():
    with pytest.raises(ValueError, match="'sustitute' or 'insert'"):
        Masked(method="unknown", model=TINY_MODEL)


def test_short_text_is_augmented(augmenter):
    samples = augmenter.augment(SHORT_TEXT, num_samples=3)
    assert 1 <= len(samples) <= 3
    for sample in samples:
        assert sample != normalize(augmenter, SHORT_TEXT)
        assert augmenter.mask_token not in sample


def test_long_text_is_augmented_in_chunks(augmenter):
    n_tokens = len(augmenter.tokenizer.tokenize(LONG_TEXT))
    assert n_tokens > augmenter.tokenizer.model_max_length

    [sample] = augmenter.augment(LONG_TEXT, num_samples=1)
    assert "##" not in sample
    assert augmenter.mask_token not in sample
    assert len(sample.split()) >= 0.8 * len(LONG_TEXT.split())


@pytest.mark.parametrize("method", METHODS)
def test_spanish_model_produces_valid_words(method):
    augmenter = Masked(
        method=method, model=SPANISH_MODEL, aug_percent=0.5, device="cpu"
    )
    [sample] = augmenter.augment(PARAGRAPH, num_samples=1)
    assert sample != PARAGRAPH
    assert "[UNK]" not in sample
    assert augmenter.mask_token not in sample
