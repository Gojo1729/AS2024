import numpy as np
import pandas as pd
from tqdm import tqdm
import nltk  # just for tokenization
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import random
import string

labels = np.array(
    ["Joy", "Trust", "Fear", "Surprise", "Sadness", "Disgust", "Anger", "Anticipation"]
)


def precision(y_true, y_pred, num_classes):
    # Initialize arrays to store true positives, false positives, and precision
    TP = np.zeros(num_classes)
    FP = np.zeros(num_classes)
    precision_scores = np.zeros(num_classes)

    # Calculate true positives and false positives for each class
    for i in range(num_classes):
        TP[i] = np.sum((y_true == i) & (y_pred == i))
        FP[i] = np.sum((y_true != i) & (y_pred == i))

    # Compute precision for each class
    for i in range(num_classes):
        if TP[i] + FP[i] > 0:
            precision_scores[i] = TP[i] / (TP[i] + FP[i])

    return np.mean(precision_scores)


def hamming_loss(y_true, y_pred):
    # Calculate number of mismatches
    num_mismatches = np.sum(y_true != y_pred)

    # Compute Hamming Loss
    hamming_loss = num_mismatches / (y_true.shape[0] * y_true.shape[1])

    return hamming_loss


def top3_accuracy(predicted_probs, true_labels):

    sorted_indices = np.argsort(predicted_probs, axis=1)[:, ::-1]

    # Check if true labels are in top-3 predicted labels
    top3_correct = np.any(
        true_labels[np.arange(len(true_labels))[:, None], sorted_indices[:, :3]], axis=1
    )
    # Calculate top-3 accuracy
    top3_accuracy = np.mean(top3_correct)

    return top3_accuracy


def tokenize(lyric: str) -> list[str]:
    # lowercase the text, remove stop words, punctuation and keep only the words
    lyric.replace("<br>", "\n")
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
