import numpy as np
import random
from pathlib import Path
from . import utils
from typing_extensions import Any

class EMBClassifier:
    def __init__(self):
        # seed for reproductiability
        random.seed(42)
        #these are the embeddings created in notebook/embeddings_approach.ipynb
        emb_path = (
            Path(Path.cwd())
            / "sentiment_classifier/embeddings/emotions_embeddings_v1.npy"
        )
        self.embeddings = np.load(emb_path)
        self.labels = np.array(
            [
                "Calmness",
                "Sadness",
                "Power",
                "Tension",
                "Amazement",
                "Solemnity",
                "Tenderness",
                "Joyful activation",
                "Nostalgia",
            ]
        )

    def cosine_similarity(
        self, embedding_matrix: np.ndarray, test_vector: np.ndarray
    ) -> np.ndarray:
        """
        Computes the cosine similarity between an embedding matrix and a test vector.

        Args:
        - embedding_matrix (np.ndarray): The embedding matrix containing multiple vectors.
        - test_vector (np.ndarray): The test vector for which cosine similarity is calculated.

        Returns:
        - np.ndarray: An array containing cosine similarity values between the test vector and each vector in the embedding matrix.
        """        
        dot_product = np.dot(embedding_matrix, test_vector)
        # Compute magnitudes
        embedding_magnitudes = np.linalg.norm(embedding_matrix, axis=1)
        test_vector_magnitude = np.linalg.norm(test_vector)
        # Compute cosine similarity
        cosine_similarity = dot_product / (
            embedding_magnitudes * test_vector_magnitude
        )
        return cosine_similarity


    def predict(self, input_lyrics: str, wv_from_bin: Any | str) -> np.ndarray:
        """
        Predicts the top 3 classes for the given input lyrics using cosine distance.

        Args:
        - input_lyrics (str): The input lyrics for prediction.
        - wv_from_bin (np.ndarray): Word embeddings loaded from a binary file.

        Returns:
        - np.ndarray: An array containing the predicted top 3 classes.
        """

        #vectorize input lyrics
        lyric_vector = utils.vectorise(wv_from_bin, input_lyrics)
        
        #compute cosine similarity
        cs = self.cosine_similarity(
           test_vector=lyric_vector, embedding_matrix=self.embeddings 
        )

        # get top 3 classes, more the similarity score, closer the embeddings
        predicted_classes = np.array(self.labels[(-cs).argsort()[:3]])

        return predicted_classes
