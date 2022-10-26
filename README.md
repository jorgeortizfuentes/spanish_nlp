# Spanish NLP

A library for Natural Language Processing in Spanish.

## Installation

```bash
pip install https://github.com/jorgeortizfuentes/spanish_nlp
```

## Usage

```python
import spacy

nlp = spacy.load("es_core_news_sm")
doc = nlp("Esta es una oración.")

for token in doc:
    print(token.text, token.pos_, token.dep_)
```

## License

This project is licensed under GNU General Public License v3.0.