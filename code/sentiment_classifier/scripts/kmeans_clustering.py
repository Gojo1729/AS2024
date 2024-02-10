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
            0: ["Amazement", "Solemnity", "Tenderness", "Calmness"],
            1: ["Joyful activation", "Power", "Tension", "Amazement"],
            2: ["Nostalgia", "Calmness", "Sadness", "Amazement"],
            3: ["Calmness", "Sadness", "Nostalgia", "Amazement"],
            4: ["Calmness", "Sadness", "Nostalgia", "Amazement"],
            5: ["Joyful activation", "Power", "Tension", "Amazement"],
            6: ["Amazement", "Solemnity", "Tenderness", "Calmness"],
            7: ["Calmness", "Sadness", "Nostalgia", "Amazement"],
            8: ["Calmness", "Sadness", "Nostalgia", "Amazement"],
        }

    def predict(self, input_lyrics: str, wv_from_bin: np.ndarray) -> list[str]:

        lyrics_vector = utils.vectorise(wv_from_bin, input_lyrics)[None, :]
        predicted_classes = self.labels[
            np.argmin(utils.cosine_distance(X=lyrics_vector, centroids=self.centroids))
        ]
        # get top 3 classes

        return predicted_classes
