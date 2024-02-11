from IPython import embed
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


def tokenize(lyric: str) -> list[str]:
    """
    Tokenizes the input lyric by lowercasing the text, removing stop words, punctuation,
    and lemmatizing the tokens.
    Args:
    - lyric (str): The input lyric to tokenize.
    Returns:
    - list[str]: A list of alpha tokens after lemmatization.
    """
    # Lowercase the text and tokenize
    tokens = nltk.tokenize.word_tokenize(lyric.lower())
    # Define stop words and punctuation
    stop_words = stopwords.words("english") + list(string.punctuation)
    # Initialize lemmatizer
    lemmatizer = WordNetLemmatizer()
    # Lemmatize tokens and filter out non-alphabetic tokens and stop words
    alpha_tokens = [
        lemmatizer.lemmatize(token)
        for token in tokens
        if token.isalpha() and token not in stop_words
    ]
    return alpha_tokens


def cosine_distance(embeddings, lyric_vector):
    # Compute cosine similarity between each data point and each centroid
    dot_product = np.dot(lyric_vector, embeddings.T)
    norms_X = np.linalg.norm(lyric_vector, axis=1)[:, np.newaxis]
    norms_centroids = np.linalg.norm(embeddings, axis=1)
    cosine_similarities = dot_product / (norms_X * norms_centroids)

    # Convert cosine similarities to cosine distances
    cosine_distances = 1 - cosine_similarities

    return cosine_distances

def vectorise(wv_from_bin, lyrics: str) -> np.ndarray:
    """
    Vectorizes the input lyrics using word embeddings.
    Args:
    - wv_from_bin: Word embeddings loaded from a binary file.
    - lyrics (str): The input lyrics to vectorize.
    Returns:
    - np.ndarray: A vector representation of the input lyrics.
    """
    #Tokenize the input lyrics
    tokens = tokenize(lyrics)
    lyric_vector = np.zeros(300)
    #Convert each token to word vector and sum it with other word vectors
    for token in tokens:
        try:
            lyric_vector += wv_from_bin.get_vector(token.lower())
        except:
            continue
    #normalize the lyric vector
    return lyric_vector / np.linalg.norm(lyric_vector)