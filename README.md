# Spanish NLP

A Python library for Natural Language Processing in Spanish.


Spanish NLP is a Python library designed for Natural Language Processing tasks in Spanish. It provides three main modules:

- preprocess: This module offers several text preprocessing options to clean and prepare texts for further analysis.
- classify: The classify module allows users to classify texts using different models and algorithms.
- augmentation: The augmentation module can be used to generate synthetic data to increase the amount of labeled data available for training models.

This project was developed by [Jorge Ortiz-Fuentes](https://ortizfuentes.com/), Linguist and Data Scientist from Chile.

## Installation

Spanish NLP can be installed via pip:

```bash
pip install spanish_nlp
```

## Usage

### Preprocessing

To preprocess text using the preprocess module, you can import it and call the desired parameters:

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

### Classification

#### Available classifiers

* Hate Speech (hate_speech)
* Toxic Speech (toxic_speech)
* Sentiment Analysis (sentiment_analysis)
* Emotion Analysis (emotion_analysis)
* Irony Analysis (irony_analysis)
* Sexist Analysis (sexist_analysis)
* Racism Analysis (racism_analysis)

#### Classification Examples

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

### Augmentation

#### Available Augmentation Models

- Spelling augmentation
  - Keyboard method
  - OCR method
  - Random method
  - Misspelling method
  - All method
- Masked augmentation
  - Sustitute method
  - Insert method
- Others models under development (such as Synonyms, WordEmbeddings, GenerativeOpenSource, GenerativeOpenAI, BackTranslation, AbstractiveSummarization)


#### Augmentation Models Examples

```python
from spanish_nlp import augmentation
ocr = augmentation.Spelling(method="ocr",
                            stopwords="default",
                            aug_percent=0.3,
                            tokenizer="default")

misspelling = augmentation.Spelling(method="keyboard",
                                    stopwords="default",
                                    aug_percent=0.3,
                                    tokenizer="default")

masked_sustitute = augmentation.Masked(method="sustitute",
                                       model="dccuchile/bert-base-spanish-wwm-cased",
                                       tokenizer="default",
                                       stopwords="default",
                                       aug_percent=0.4,
                                       device="cpu",
                                       top_k=10)

masked_insert = augmentation.Masked(method="insert",
                                    model="dccuchile/bert-base-spanish-wwm-cased",
                                    tokenizer="default",
                                    stopwords="default",
                                    aug_percent=0.4,
                                    device="cpu",
                                    top_k=10)

text = "En aquel tiempo yo tenía veinte años y estaba loco. Había perdido un país pero había ganado un sueño. Y si tenía ese sueño lo demás no importaba. Ni trabajar ni rezar ni estudiar en la madrugada junto a los perros románticos."

new_texts = [text]
new_texts.append(ocr.augment(text, num_samples=1, num_workers=1))
new_texts.append(misspelling.augment(text, num_samples=1, num_workers=1))
new_texts.append(masked_sustitute.augment(text, num_samples=1))
new_texts.append(masked_insert.augment(text, num_samples=1))

for t in new_texts:
    print(t)
    print("---")

```

Output:

```bash
En aquel tiempo yo tenía veinte años y estaba loco. Había perdido un país pero había ganado un sueño. Y si tenía ese sueño lo demás no importaba. Ni trabajar ni rezar ni estudiar en la madrugada junto a los perros románticos.
---
['3n aqueI 7iempo yo t3nía veinte añQs V 3sta8a loGo. Había perO10o un país pero había Canado un BueñQ. V si t3nía ese su3N0 lo d3WáB no imp0rtaEa. Hi trabaLar ni rezaP ni estudiaP en la maOPuga0a Lun7o a IoB perros roWánticos.']
---
['En squel tjempo yo tfbíx vsknte alod y estxba lpfo. Hanía pfddido un país pero hqvís ganaeo uj skeol. Y si tebía ese syrño lo demáz no jmppfgabx. Nj travayar ni rezar mu estudist eh la nadtugads junto a loa peerks eomábticox.']
---
['En aquel tiempo yo tenía 18 años y estaba loco. Había arruinado un hogar pero había ganado un sueño. Ahora si tenía ese sueño lo demás no importaba. Pero trabajar ni rezar ni trabajar en la madrugada junto a los perros ni']
---
['En aquel tiempo yo tenía los veinteséis años y estaba loco. Había perdido un gran país pero sí había ganado tener un sueño. Y si tenía ese sueño lo demás ya no importaba.. Ni trabajar ni rezar ni estudiar en la madrugada junto a los perros románticos.']
---
```
## License

Spanish NLP is licensed under the [LICENSE](GNU General Public License v3.0).

## Contributing and roadmap

Contributions to Spanish NLP are welcome! Please see the [ROADMAP.md](contributing guide) for more information.

## Acknowledgements

We would like to express our gratitude to the Millennium Institute For Foundational Research and Department of Computer Science at the University of Chile for supporting the development of Spanish NLP. Special thanks to Felipe Bravo-Marquéz, Ricardo Cordova and Hernán Sarmiento for their knowledge, support and invaluable contribution to the project.