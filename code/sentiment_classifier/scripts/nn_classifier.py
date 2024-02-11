import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
import nltk  # just for tokenization
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import random
from pathlib import Path
from . import utils
from typing_extensions import Any


class NN:
    def __init__(self) -> None:
        random.seed(42)
        embeddings_path = Path(Path.cwd()) / "code/sentiment_classifier/embeddings"
        with open(embeddings_path / "nn.pickle", "rb") as f:
            self.nn = pickle.load(f)
        self.labels = np.array(
            ['Joy', 'Trust', 'Fear', 'Surprise','Sadness', 'Disgust', 'Anger', 'Anticipation']
        )

    def predict(self, input_lyrics: str, wv_from_bin: Any | str) -> np.ndarray:
        """
        Predicts the top 3 emotion labels for the input lyrics.

        Args:
        - input_lyrics (str): Input lyrics for prediction.
        - wv_from_bin (np.ndarray): Word vectors for the lyrics.

        Returns:
        - np.ndarray: Top 3 predicted emotion labels.
        """
        lyrics_vector = utils.vectorise(wv_from_bin, input_lyrics)[None, :]
        probs = self.nn.predict(lyrics_vector)

        #use argsort, reverse it as argsort provides indexes in ascending order but we need the indexes
        # of the probabilities in descending order, we choose classes with high probability scores. 
        predictions = np.array(self.labels)[np.argsort(probs[0])[::-1]]
        # get top 3 classes
        return predictions[:3]
