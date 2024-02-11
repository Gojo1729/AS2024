import numpy as np
import random
from pathlib import Path
from . import utils
from typing_extensions import Any


class KMeans:
    def __init__(self) -> None:
        """
        Initializes the KMeans classifier.

        Loads the precomputed centroids and assigns labels to each centroid.

        """
        random.seed(42)
        p = (
            Path(Path.cwd())
            / "sentiment_classifier/embeddings/kmeans_centeroids_v1.npy"
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

    def predict(self, input_lyrics: str, wv_from_bin: Any | str) -> list[str]:
        """
        Predicts the sentiment categories for input lyrics using KMeans clustering.

        Args:
        - input_lyrics (str): The input lyrics to predict the sentiment categories for.
        - wv_from_bin (np.ndarray): Word embeddings loaded from a binary file.

        Returns:
        - list[str]: The predicted sentiment categories for the input lyrics.

        """        

        lyrics_vector = utils.vectorise(wv_from_bin, input_lyrics)[None, :]
        # Get the labels corresponding to the closest centroid
        predicted_classes = self.labels[
            np.argmin(utils.cosine_distance(lyric_vector=lyrics_vector, embeddings=self.centroids))
        ]
        return predicted_classes
