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
        remove_hashtags=False,
        split_hashtags=True,
        normalize_breaklines=True,
        remove_emoticons=True,
        remove_emojis=True,
        convert_emoticons=False,
        convert_emojis=False,
        normalize_inclusive_language=False,
        reduce_spam=True,
        remove_vowels_accents=True,
        remove_multiple_spaces=True,
        remove_punctuation=True,
        remove_unprintable=True,
        remove_numbers=True,
        remove_stopwords=False,
        stopwords_list=None,
        lemmatize=False,
        stem=False,
        remove_html_tags=True,
)

test_text = """𝓣𝓮𝔁𝓽𝓸 𝓭𝓮 𝓹𝓻𝓾𝓮𝓫𝓪

<b>Hola </b>, este es un texto de prueba :) a continuación les mostraré un poema de Roberto Bolaño llamado "Los perros románticos" 🤭👀😅

https://www.poesi.as/rb9301.htm

Me gusta la LINGÜÍSTICA y los ñandúes… También los pingüinos 🐧🐧🐧. #VivanLosPinguinos #SíSeñor #PinguinosDelMundoUníos #ÑanduesDelMundoTambién

Si colaboras con este código te puedes ganar $10.000.000.000. O tal vez 2000 vacas. Mi teléfono es +569123456789"""

print(sp.transform(test_text, debug=False))

```

Output: 
```bash
hola este es un texto de prueba a continuacion les mostrare un poema de roberto bolaño llamado los perros romanticos 
me gusta la linguistica y los ñandues tambien los pinguinos vivan los pinguinos si señor pinguinos del mundo unios ñandues del mundo tambien
si colaboras con este codigo te puedes ganar o tal vez vacas mi telefono es
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
t1 =  "LAS RATAS QUE ESTÁN EN EL CONGRESO DEBERÍAN SER EXTERMINADAS"
t2 = "El presidente convocó a una reunión a los representantes de los partidos políticos"
p1 = sc.predict(t1)
p2 = sc.predict(t2)

print("Text 1: ", t1)
print("Prediction 1: ", p1)
print("Text 2: ", t2)
print("Prediction 2: ", p2)
```

Output:
```bash
Text 1:  LAS RATAS QUE ESTÁN EN EL CONGRESO DEBERÍAN SER EXTERMINADAS
Prediction 1:  {'hateful': 0.29868438839912415, 'aggressive': 0.1646653413772583, 'targeted': 0.0075755491852760315}
Text 2:  El presidente convocó a una reunión a los representantes de los partidos políticos
Prediction 2:  {'targeted': 0.013353983871638775, 'aggressive': 0.010659483261406422, 'hateful': 0.009115356020629406}
```

## Pending:

* Include better documentation
* Include information about available classifiers

## License

This project is licensed under GNU General Public License v3.0.

