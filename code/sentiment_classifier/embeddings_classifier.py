import numpy as np
import nltk  # just for tokenization
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import random
import string
import pathlib
from pathlib import Path
import os


class EMBClassifier:
    def __init__(self):
        random.seed(42)
        print(Path.cwd())
        p = (
            Path(Path.cwd())
            / "code/sentiment_classifier/embeddings/emotions_embeddings_v1.npy"
        )
        self.embeddings = np.load(p)
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

    def tokenize(self, lyric: str) -> list[str]:
        # lowercase the text, remove stop words, punctuation and keep only the words
        tokens = nltk.tokenize.word_tokenize(lyric.lower())
        stop_words = stopwords.words("english") + list(string.punctuation)
        lemmatizer = WordNetLemmatizer()
        alpha_tokens = [
            lemmatizer.lemmatize(token)
            for token in tokens
            if token.isalpha() and token not in stop_words
        ]

        return alpha_tokens

    def vectorise(self, wv_from_bin, lyrics: str) -> np.ndarray:
        tokens = self.tokenize(lyrics)
        lyric_vector = np.zeros(300)
        for token in tokens:
            try:
                lyric_vector += wv_from_bin.get_vector(token.lower())
            except:
                continue
        return lyric_vector / np.linalg.norm(lyric_vector)

    def distance(
        self, metric: str, embedding_matrix: np.ndarray, test_vector: np.ndarray
    ) -> np.ndarray:
        if metric == "cosine":

            dot_product = np.dot(embedding_matrix, test_vector)

            # Compute magnitudes
            embedding_magnitudes = np.linalg.norm(embedding_matrix, axis=1)
            test_vector_magnitude = np.linalg.norm(test_vector)

            # Compute cosine similarity
            cosine_similarity = dot_product / (
                embedding_magnitudes * test_vector_magnitude
            )
            return cosine_similarity

        elif metric == "euclidean":
            distances = np.linalg.norm(self.embeddings - test_vector, axis=1)
            return distances
        else:
            raise Exception(f"Invalid parameter value {metric}")

    def predict(self, input_lyrics: str, wv_from_bin: np.ndarray) -> np.ndarray:

        test1_vector = self.vectorise(wv_from_bin, input_lyrics)
        cs = self.distance(
            metric="cosine", embedding_matrix=self.embeddings, test_vector=test1_vector
        )
        # get top 3 classes
        predicted_classes = np.array(self.labels[(-cs).argsort()[:3]])

        return predicted_classes
