import numpy as np
import pandas as pd
from tqdm import tqdm
import json
from pathlib import Path


class Generator:
    def __init__(self):
        p = Path(Path.cwd()) / "code/lyrics_generator"
        self.w1 = np.load(p / "w1.npy")
        self.w2 = np.load(p / "w2.npy")
        with open(p / "word_index.json", "r") as json_file:
            self.word_index = json.load(json_file)

        self.v_count = 1331
        with open(p / "index_word.json", "r") as json_file:
            self.index_word = json.load(json_file)

    def word_sim(self, word, top_n):

        w1_index = self.word_index[word]
        v_w1 = self.w1[w1_index]

        # CYCLE THROUGH VOCAB
        word_sim = {}
        for i in range(self.v_count):
            v_w2 = self.w1[i]
            theta_num = np.dot(v_w1, v_w2)
            theta_den = np.linalg.norm(v_w1) * np.linalg.norm(v_w2)
            theta = theta_num / theta_den

            word = self.index_word[str(i)]
            word_sim[word] = theta

        words_sorted = sorted(word_sim.items(), key=lambda item: item[1], reverse=True)

        words = []
        for word in words_sorted[:top_n]:
            words.append(word[0])

        return words

    def generate_lyrics(self, mood: str):
        n_words = 100
        previous_word = initial_token = "i"
        predicted_lyrics = initial_token
        previous_five_words = [initial_token]

        for _ in tqdm(range(n_words)):
            count = 0
            tentaitve_next_words = self.word_sim(previous_word, 10)

            tentaitve_next_word = np.random.choice(tentaitve_next_words + [",", "."])
            if tentaitve_next_word in [",", "."]:
                previous_five_words.append(tentaitve_next_word)
                predicted_lyrics += f" {tentaitve_next_word}"

            else:
                while count < 10 and (tentaitve_next_word in previous_five_words):
                    tentaitve_next_word = tentaitve_next_words[count]
                    count += 1

                predicted_lyrics += f" {tentaitve_next_word}"
                if len(previous_five_words) == 5:
                    previous_five_words.pop(0)

                previous_word = tentaitve_next_word
                previous_five_words.append(previous_word)

        return predicted_lyrics
