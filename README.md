# Spanish NLP

A library for Natural Language Processing in Spanish.

## Installation

```bash
pip install git+https://github.com/jorgeortizfuentes/spanish_nlp
```

## Preprocess usage

```python

from spanish_nlp import preprocess
sp = preprocess.SpanishPreprocess(
    lower=True,
    remove_url=True,
    remove_hashtags=True,
    preserve_emoticons=True,
    preserve_emojis=True,
    convert_emoticons=False,
    convert_emojis=False,
    normalize_inclusive_language=True,
    reduce_spam=True,
    remove_vowels_accents=True,
    remove_punctuation=True,
    remove_unprintable=True,
    remove_numbers=True,
    remove_stopwords=False,
    stopwords_list=None,
    stem=False,
)

test_text = """𝓣𝓮𝔁𝓽𝓸 𝓭𝓮 𝓹𝓻𝓾𝓮𝓫𝓪

Este es un texto de prueba :) a continuación les mostraré un poema llamado "Los perros románticos" 🤭👀😅

https://www.poesi.as/rb9301.htm

Me gusta la LINGÜÍSTICA y los ñandúes… También los pingüinos 🐧🐧🐧. 

Tengo una deuda de $10.000.000.000, pero tengo 2000 vacas. Mi teléfono es +5698791045"""

print(sp.transform(test_text))

```

## Classification usage

### Available classifiers

* Hate Speech (hate_speech)
* Toxic Speech (toxic_speech)
* Sentiment Analysis (sentiment_analysis)
* Emotion Analysis (emotion_analysis)
* Irony Analysis (irony_analysis)
* Sexist Analysis (sexist_analysis)
* Racism Analysis (racism_analysis)

### Example

```python
from spanish_nlp import classifiers

sc = classifiers.SpanishClassifier(model_name="hate_speech", device='cpu')
t1 = "ODIO LA POLÍTICA Y A LAS RATAS QUE ESTÁN EN EL CONGRESO DEBERÍAN SER EXTERMINADAS"
t2 = "El presidente convocó a una reunión a los representantes de los partidos políticos"
p1 = sc.predict(t1)
p2 = sc.predict(t2)

print("Text 1: ", t1)
print("Prediction 1: ", p1)
print("Text 2: ", t2)
print("Prediction 2: ", p2)
```

## Pending:

* Add default stopwords lists
* Generate documentation
* Add information about available classifiers

## License

This project is licensed under GNU General Public License v3.0.

