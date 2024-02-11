# class to be imported in the code/model_interface while loading the pickle file
import numpy as np
from collections import defaultdict
from tqdm import tqdm

class word2vec:
    def __init__(self):
        self.n = 3
        self.eta = 1e-2
        self.epochs = 5
        self.window = 3
        self.emotion = "Joy"

    def generate_training_data(self, settings, corpus):
        """
        Generate training data for the Word2Vec model.

        Args:
        - settings (dict): Settings for generating training data.
        - corpus (list): List of sentences in the corpus.

        Returns:
        - list: Training data containing word pairs.
        """
        word_counts = defaultdict(int)
        for row in corpus:
            for word in row:
                word_counts[word] += 1

        self.v_count = len(word_counts.keys())

        self.words_list = sorted(list(word_counts.keys()), reverse=False)
        self.word_index = dict((word, i) for i, word in enumerate(self.words_list))
        self.index_word = dict((i, word) for i, word in enumerate(self.words_list))

        training_data = []
        for sentence in tqdm(corpus):
            sent_len = len(sentence)
            for i, word in enumerate(sentence):
                w_target = self.word2onehot(sentence[i])
                w_context = []
                for j in range(i - self.window, i + self.window + 1):
                    if j != i and j <= sent_len - 1 and j >= 0:
                        w_context.append(self.word2onehot(sentence[j]))
                training_data.append([w_target, w_context])

        return training_data

    def softmax(self, x):
        """
        Compute the softmax function.

        Args:
        - x (np.ndarray): Input array.

        Returns:
        - np.ndarray: Output array after applying softmax.
        """
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def word2onehot(self, word):
        """
        Convert a word to its one-hot encoding representation.

        Args:
        - word (str): Input word.

        Returns:
        - list: One-hot encoded representation of the word.
        """
        word_vec = [0 for i in range(0, self.v_count)]
        word_index = self.word_index[word]
        word_vec[word_index] = 1
        return word_vec

    def forward_pass(self, x):
        """
        Perform a forward pass in the Word2Vec model.

        Args:
        - x (np.ndarray): Input vector.

        Returns:
        - tuple: Output vector, hidden layer, and unnormalized output.
        """
        h = np.dot(self.w1.T, x)
        u = np.dot(self.w2.T, h)
        y_c = self.softmax(u)
        return y_c, h, u

    def backprop(self, e, h, x):
        """
        Perform backpropagation in the Word2Vec model.

        Args:
        - e (np.ndarray): Error vector.
        - h (np.ndarray): Hidden layer output.
        - x (np.ndarray): Input vector.
        """    
        dl_dw2 = np.outer(h, e)
        dl_dw1 = np.outer(x, np.dot(self.w2, e.T))
        self.w1 = self.w1 - (self.eta * dl_dw1)
        self.w2 = self.w2 - (self.eta * dl_dw2)

    def train(self, training_data):
        """
        Train the Word2Vec model.

        Args:
        - training_data (list): Training data containing word pairs.
        """    
        limit1 = np.sqrt(2 / float(self.n + self.v_count))
        self.w1 = np.random.normal(
            0.0, limit1, size=(self.v_count, self.n)
        )  # embedding matrix
        self.w2 = np.random.normal(0.0, limit1, size=(self.n, self.v_count))

        for i in tqdm(range(0, self.epochs)):
            self.loss = 0
            for w_t, w_c in training_data:
                y_pred, h, u = self.forward_pass(w_t)
                EI = np.sum([np.subtract(y_pred, word) for word in w_c], axis=0)
                self.backprop(EI, h, w_t)
                self.loss += -np.sum([u[word.index(1)] for word in w_c]) + len(
                    w_c
                ) * np.log(np.sum(np.exp(u)))

            print("EPOCH:", i, "LOSS:", self.loss)

    def word_vec(self, word):
        w_index = self.word_index[word]
        v_w = self.w1[w_index]
        return v_w

    def vec_sim(self, vec, top_n):

        word_sim = {}
        for i in range(self.v_count):
            v_w2 = self.w1[i]
            theta_num = np.dot(vec, v_w2)
            theta_den = np.linalg.norm(vec) * np.linalg.norm(v_w2)
            theta = theta_num / theta_den

            word = self.index_word[i]
            word_sim[word] = theta

        words_sorted = sorted(word_sim.items(), key=lambda item: item[1], reverse=True)

        return words_sorted

    def word_sim(self, word, top_n):
        w1_index = self.word_index[word]
        v_w1 = self.w1[w1_index]
        word_sim = {}
        for i in range(self.v_count):
            v_w2 = self.w1[i]
            theta_num = np.dot(v_w1, v_w2)
            theta_den = np.linalg.norm(v_w1) * np.linalg.norm(v_w2)
            theta = theta_num / theta_den

            word = self.index_word[i]
            word_sim[word] = theta

        words_sorted = sorted(word_sim.items(), key=lambda item: item[1], reverse=True)
        words = []
        for word in words_sorted[:top_n]:
            words.append(word[0])

        return words
