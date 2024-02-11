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


class KMeans:
    def __init__(self) -> None:
        random.seed(42)
        p = (
            Path(Path.cwd())
            / "code/sentiment_classifier/embeddings/kmeans_centeroids_v1.npy"
        )
        self.centroids = np.load(p)
        self.labels = {
            0: ["Amazement", "Solemnity", "Tenderness"],
            1: ["Joyful activation", "Power", "Tension"],
            2: ["Calmness", "Sadness", "Nostalgia"],
            3: ["Amazement", "Solemnity", "Tenderness"],
            4: ["Calmness", "Sadness", "Amazement"],
            5: ["Calmness", "Sadness", "Nostalgia"],
            6: ["Calmness", "Sadness", "Amazement"],
            7: ["Joyful activation", "Power", "Tension"],
            8: ["Calmness", "Sadness", "Nostalgia"],
        }

    def predict(self, input_lyrics: str, wv_from_bin: np.ndarray) -> list[str]:

        lyrics_vector = utils.vectorise(wv_from_bin, input_lyrics)[None, :]
        predicted_classes = self.labels[
            np.argmin(utils.cosine_distance(X=lyrics_vector, centroids=self.centroids))
        ]
        # get top 3 classes

        return predicted_classes
