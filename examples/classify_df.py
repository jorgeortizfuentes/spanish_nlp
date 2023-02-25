from datetime import datetime

import pandas as pd

from spanish_nlp.classifiers import SpanishClassifier
from spanish_nlp.preprocess import SpanishPreprocess

# Load Spanish DataFrame with tweets
df = pd.read_csv(
    "tweets.csv",
    sep=";",
    encoding="utf-8",
    quotechar='"',
    dtype={"id": object, "user_id": object},
)

sp = SpanishPreprocess(
    lower=True,
    remove_url=True,
    remove_hashtags=True,
    preserve_emojis=True,
    preserve_emoticons=True,
    convert_emoticons=False,
    convert_emojis=False,
    normalize_inclusive_language=False,
    reduce_spam=True,
    remove_vowels_accents=True,
    remove_punctuation=True,
    remove_unprintable=True,
    remove_numbers=True,
    remove_stopwords=False,
    stopwords_list=None,
    stem=False,
)


df["text"] = df["text"].swifter.apply(sp.transform)

df = df[df.text.notnull()]
df = df[df.text != ""]
df = df[df["text"].apply(lambda x: type(x) == str)]
df = df.reset_index(drop=True)


def predict_label(text, model, file_log):
    try:
        return model.predict(text)
    except Exception as e:
        time = datetime.now().strftime("%d-%Y-%m %H:%M:%S")
        # Write log
        with open(file_log, "a") as f:
            f.write(f"{time}. Error: {e}\n")
            f.write(f"{model}. Model: {model}\n")
            f.write(f"{time}. Text: {text}\n")
        return None


classifiers_names = [
    "hate_speech",
    "toxic_speech",
    "sentiment_analysis",
    "emotion_analysis",
    "irony_analysis",
    "sexist_analysis",
    "racism_analysis",
]
classifiers = {}

file_log = "classification.log"

classifiers_names = [
    "hate_speech",
    "toxic_speech",
    "sentiment_analysis",
    "emotion_analysis",
    "irony_analysis",
    "sexist_analysis",
    "racism_analysis",
]
classifiers = {}

for n in classifiers_names:
    classifiers[n] = SpanishClassifier(model_name=n, device=0)

for cl_name in classifiers.keys():
    df[cl_name] = None
    df[cl_name] = df["text"].swifter.apply(
        lambda x: predict_label(x, classifiers[cl_name], file_log)
    )
    df.to_pickle("tweets_classified.pkl")
