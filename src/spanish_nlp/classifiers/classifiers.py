import torch
from transformers import pipeline


class SpanishClassifier:
    def __init__(self, model_name=None, device=None):
        self.model_name = model_name
        self.device = device
        if self.device is None:
            self._set_default_device_()
        self.model = NotImplemented
        self.labels = NotImplemented
        self.last_prediction = NotImplemented
        self.multiclass = NotImplemented
        if self.model_name is not None:
            self._configure_model_()

    def _set_default_device_(self):
        self.device = 0 if torch.cuda.is_available() else -1

    def set_model(self, model_name):
        self.model_name = model_name

    def _configure_model_(self):
        if self.model_name == "hate_speech":
            self.load_hate_speech()
        elif self.model_name == "incivility":
            self.load_incivility()
        elif self.model_name == "toxic_speech":
            self.load_toxic_speach()
        elif self.model_name == "sentiment_analysis":
            self.load_sentiment_analysis()
        elif self.model_name == "emotion_analysis":
            self.load_emotion_analysis()
        elif self.model_name == "irony_analysis":
            self.load_irony_analysis()
        elif self.model_name == "sexist_analysis":
            self.load_sexist_analysis()
        elif self.model_name == "racism_analysis":
            self.load_racism_analysis()

    def get_info_about_models(self):
        info = {
            "hate_speech": {
                "function": self.load_hate_speech(),
                "types": ["robertuito", "bert"],
            },
            "incivility": {
                "function": self.load_incivility(),
                "types": ["bert"],
            },
            "toxic_speech": {
                "function": self.load_toxic_speach(),
                "types": ["political-tweets-es"],
            },
            "sentiment_analysis": {
                "function": self.load_sentiment_analysis(),
                "types": ["robertuito"],
            },
            "emotion_analysis": {
                "function": self.load_emotion_analysis(),
                "types": ["robertuito"],
            },
            "irony_analysis": {
                "function": self.load_irony_analysis(),
                "types": ["robertuito"],
            },
            "sexist_analysis": {
                "function": self.load_sexist_analysis(),
                "types": ["sexist_analysis_metwo"],
            },
            "racist_analysis": {
                "function": self.load_racism_analysis(),
                "types": ["racism_paula_lobo_et_al_average_strict"],
            },
        }
        return info

    def load_hate_speech(self, type="bert"):
        if type == "robertuito":
            self._robertuito_hate_speech_()
        elif type == "bert":
            self._bert_hate_speech_()
        else:
            raise ValueError("Unknown type of hate speech model")

    def _robertuito_hate_speech_(self):
        self.model = pipeline(
            "text-classification",
            model="pysentimiento/robertuito-hate-speech",
            truncation=True,
            max_length=128,
            device=self.device,
        )
        self.max_length = 128
        self.type_model = "hf"
        self.n_labels = 3
        self.multiclass = True
        self.labels = {
            "hateful": "hateful",
            "aggressive": "aggressive",
            "targeted": "targeted",
        }

    def _bert_hate_speech_(self):
        self.model = pipeline(
            "text-classification",
            model="jorgeortizfuentes/spanish_hate_speech",
            truncation=True,
            max_length=512,
            device=self.device,
        )
        self.max_length = 512
        self.type_model = "hf"
        self.n_labels = 2
        self.multiclass = False
        self.labels = {
            "hate": "hate",
            "no_hate": "no_hate",
        }

    def load_incivility(self, type="bert"):
        if type == "bert":
            self._bert_incivility_()
        else:
            raise ValueError("Unknown type of hate speech model")

    def _bert_incivility_(self):
        self.model = pipeline(
            "text-classification",
            model="jorgeortizfuentes/spanish_incivility",
            truncation=True,
            max_length=512,
            device=self.device,
        )
        self.max_length = 512
        self.type_model = "hf"
        self.n_labels = 2
        self.multiclass = False
        self.labels = {
            "incivility": "incivility",
            "no_incivility": "no_incivility",
        }

    def load_toxic_speach(self, type="political-tweets-es"):
        if type == "political-tweets-es":
            self._toxic_political_tweets_()
        else:
            raise ValueError("Unknown type of toxic speech model")

    def _toxic_political_tweets_(self):
        self.model = pipeline(
            "text-classification",
            model="Newtral/xlm-r-finetuned-toxic-political-tweets-es",
            truncation=True,
            max_length=512,
            device=self.device,
        )
        self.max_length = 512
        self.type_model = "hf"
        self.multiclass = False
        self.n_labels = 2
        self.labels = {
            "LABEL_0": "toxic",
            "LABEL_1": "very_toxic",
        }

    def load_sentiment_analysis(self, type="robertuito"):
        if type == "robertuito":
            self._robertuito_sentiment_analysis_()
        else:
            raise ValueError("Unknown type of sentiment analysis model")

    def _robertuito_sentiment_analysis_(self):
        self.model = pipeline(
            "text-classification",
            model="pysentimiento/robertuito-sentiment-analysis",
            truncation=True,
            max_length=128,
            device=self.device,
        )
        self.max_length = 128
        self.type_model = "hf"
        self.multiclass = False
        self.n_labels = 3
        self.labels = {"NEU": "neutral", "NEG": "negative", "POS": "positive"}

    def load_emotion_analysis(self, type="robertuito"):
        if type == "robertuito":
            self._robertuito_emotion_analysis_()
        else:
            raise ValueError("Unknown type of sentiment analysis model")

    def _robertuito_emotion_analysis_(self):
        self.model = pipeline(
            "text-classification",
            model="pysentimiento/robertuito-emotion-analysis",
            truncation=True,
            max_length=128,
            device=self.device,
        )
        self.max_length = 128
        self.type_model = "hf"
        self.multiclass = False
        self.n_labels = 7
        self.labels = {
            "others": "others",
            "joy": "joy",
            "sadness": "sadness",
            "anger": "anger",
            "surprise": "surprise",
            "disgust": "disgust",
            "fear": "fear",
        }

    def load_irony_analysis(self, type="robertuito"):
        if type == "robertuito":
            self._robertuito_irony_analysis_()
        else:
            raise ValueError("Unknown type of irony analysis model")

    def _robertuito_irony_analysis_(self):
        self.model = pipeline(
            "text-classification",
            model="pysentimiento/robertuito-irony",
            truncation=True,
            max_length=128,
            device=self.device,
        )
        self.max_length = 128
        self.type_model = "hf"
        self.multiclass = False
        self.n_labels = 2
        self.labels = {"not ironic": "not_ironic", "ironic": "ironic"}

    def load_sexist_analysis(self, type="sexist_analysis_metwo"):
        if type == "sexist_analysis_metwo":
            self._sexist_analysis_metwo_()
        else:
            raise ValueError("Unknown type of sexist analysis model")

    def _sexist_analysis_metwo_(self):
        self.model = pipeline(
            "text-classification",
            model="hackathon-pln-es/twitter_sexismo-finetuned-exist2021-metwo",
            truncation=True,
            max_length=128,
            device=self.device,
        )
        self.max_length = 128
        self.type_model = "hf"
        self.multiclass = False
        self.n_labels = 2
        self.labels = {"LABEL_0": "not_sexist", "LABEL_1": "sexist"}

    def load_racism_analysis(
        self, type="racism_paula_lobo_et_al_average_strict"
    ):
        if type == "racism_paula_lobo_et_al_average_strict":
            self._racism_paula_lobo_et_al_average_()
        else:
            raise ValueError("Unknown type of racism analysis model")

    def _racism_paula_lobo_et_al_average_(self):
        self.model = pipeline(
            "text-classification",
            model="MartinoMensio/racism-models-m-vote-strict-epoch-4",
            tokenizer="dccuchile/bert-base-spanish-wwm-uncased",
            truncation=True,
            max_length=512,
            device=self.device,
        )
        self.max_length = 512
        self.type_model = "hf"
        self.multiclass = False
        self.n_labels = 2
        self.labels = {"non-racist": "non-racist", "racist": "racist"}

    def _predict_hf_(self, text):
        prediction = self.model(
            text,
            top_k=self.n_labels,
            truncation=True,
            max_length=self.max_length,
        )
        d_prediction = {}
        for p in prediction:
            # p["label"] = self.labels[p["label"]]
            d_prediction[self.labels[p["label"]]] = p["score"]
        return d_prediction

    def predict(self, text):
        if self.type_model == "hf":
            if isinstance(text, str):
                self.last_prediction = self._predict_hf_(text)
                return self.last_prediction
            elif isinstance(text, list):
                self.last_prediction = []
                for t in text:
                    self.last_prediction.append(self._predict_hf_(t))
                return self.last_prediction
