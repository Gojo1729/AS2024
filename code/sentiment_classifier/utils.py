import numpy as np
import pandas as pd
from tqdm import tqdm
import nltk  # just for tokenization
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import random
import string

labels = np.array(
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


def tokenize(lyric: str) -> list[str]:
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


def cosine_distance(X, centroids):
    # Compute cosine similarity between each data point and each centroid
    dot_product = np.dot(X, centroids.T)
    norms_X = np.linalg.norm(X, axis=1)[:, np.newaxis]
    norms_centroids = np.linalg.norm(centroids, axis=1)
    cosine_similarities = dot_product / (norms_X * norms_centroids)

    # Convert cosine similarities to cosine distances
    cosine_distances = 1 - cosine_similarities

    return cosine_distances


def vectorise(wv_from_bin, lyrics: str) -> np.ndarray:
    tokens = tokenize(lyrics)
    lyric_vector = np.zeros(300)
    for token in tokens:
        try:
            lyric_vector += wv_from_bin.get_vector(token.lower())
        except:
            continue
    return lyric_vector / np.linalg.norm(lyric_vector)
