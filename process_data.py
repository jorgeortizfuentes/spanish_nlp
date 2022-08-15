import os
from datetime import datetime

import pandas as pd
import swifter
from tqdm import tqdm

from classifiers import SpanishClassifier
from preprocess import SpanishPreprocess

df_tweets = pd.read_csv(
    "data/tweets_elections_raw.csv",
    sep=";",
    encoding="utf-8",
    quotechar='"',
    dtype={"id": object, "user_id": object},
)


# Cluster 1: Boric, CLuster 2: Kast
df_tweets["source"] = "twitter"

# Only rows with language = "es"
df_tweets = df_tweets[df_tweets["language"] == "es"]

df_tweets = df_tweets.rename(columns={"tweet": "text", "cluster": "candidato"})
df_tweets = df_tweets[["user_id", "date", "text", "candidato", "source"]]
df_tweets = df_tweets.reset_index(drop=True)

# Replace candidato == 1 with "Boric" and candidato == 2 with "Kast"
df_tweets["candidato"] = df_tweets["candidato"].replace(1, "Boric")
df_tweets["candidato"] = df_tweets["candidato"].replace(2, "Kast")


df_wsp = pd.read_pickle("data/wsp_elections_raw.pkl")

df_wsp = df_wsp.rename(
    columns={"remote_resource": "user_id", "data": "text", "key_remote_jid": "group_id"}
)
df_wsp = df_wsp[["user_id", "date", "text", "candidato", "key_from_me"]]

# Remove rows with key_from_me = 1
df_wsp = df_wsp[df_wsp.key_from_me == 0]

df_wsp = df_wsp.dropna(subset=["text"])
# Remove rows with empty text or NaNs:
df_wsp = df_wsp[df_wsp.text.notnull()]
df_wsp = df_wsp[df_wsp.text != ""]
del df_wsp["key_from_me"]
df_wsp = df_wsp.reset_index(drop=True)
df_wsp["source"] = "whatsapp"


df = pd.concat([df_tweets, df_wsp], ignore_index=True)
df["n_words"] = df["text"].swifter.apply(lambda x: len(x.split()))
df = df[df["n_words"] >= 4]
del df["n_words"]
df = df.reset_index(drop=True)

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


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

tqdm.pandas(desc="Classifying texts")


tqdm.pandas(desc="Classifying texts")


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
            # time.sleep(1)
            # Delete cache GPU
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


file_log = "data/classification.log"

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
    df.to_pickle("data/data_classified.pkl")
