import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
import nltk  # just for tokenization
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import random
import string
from pathlib import Path
from . import utils


class NN:
    def __init__(self) -> None:
        random.seed(42)
        p = Path(Path.cwd()) / "code/sentiment_classifier/embeddings"
        with open(p / "nn.pickle", "rb") as f:
            self.nn = pickle.load(f)
        self.labels = np.array(
            [
                "Joy",
                "Trust",
                "Fear",
                "Surprise",
                "Sadness",
                "Disgust",
                "Anger",
                "Anticipation",
            ]
        )

    def predict(self, input_lyrics: str, wv_from_bin: np.ndarray) -> np.ndarray:

        lyrics_vector = utils.vectorise(wv_from_bin, input_lyrics)[None, :]
        probs = self.nn.predict(lyrics_vector)
        predictions = np.array(self.labels)[np.argsort(probs[0])[::-1]]
        # get top 3 classes
        return predictions[:3]
