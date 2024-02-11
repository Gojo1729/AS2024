import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
import nltk  # just for tokenization
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import random
import string
from pathlib import Path
import json


class LyricsGenerator:
    def __init__(self, emotion) -> None:
        random.seed(42)
        self.emotion = emotion
        p = Path(Path.cwd()) / f"code/lyrics_generator/embeddings/{self.emotion}"

        with open(p / "w2v.pickle", "rb") as f:
            self.w2v = pickle.load(f)

        with open(p / "template.json", "r") as file:
            self.template = json.load(file)

        self.labels = np.array(
            [
                "Joy",
                "Trust",
                "Fear",
                "Surprise",
                "Sadness",
                "Disgust",
                "Anger",
                "Anticipation",
            ]
        )

        self.pos_mapping = {
            "coordinating_conjunction": ["CC"],
            "cardinal_digit": ["CD"],
            "determiner": ["DT"],
            "existential_there": ["EX"],
            "foreign_word": ["FW"],
            "preposition_subordinating_conjunction": ["IN"],
            "adjective_large": ["JJ"],
            "adjective_larger": ["JJR"],
            "adjective_largest": ["JJS"],
            "list_market": ["LS"],
            "modal": ["MD"],
            "noun_singular": ["NN"],
            "noun_plural": ["NNS"],
            "proper_noun_singular": ["NNP"],
            "proper_noun_plural": ["NNPS"],
            "predeterminer": ["PDT"],
            "possessive_ending": ["POS"],
            "personal_pronoun": ["PRP"],
            "possessive_pronoun": ["PRP$"],
            "adverb": ["RB"],
            "adverb_comparative": ["RBR"],
            "adverb_superlative": ["RBS"],
            "particle": ["RP"],
            "infinite_marker": ["TO"],
            "interjection": ["UH"],
            "verb": ["VB"],
            "verb_gerund": ["VBG"],
            "verb_past_tense": ["VBD"],
            "verb_past_participle": ["VBN"],
            "verb_present_not_third_person_singular": ["VBP"],
            "verb_present_third_person_singular": ["VBZ"],
            "wh_determiner": ["WDT"],
            "wh_pronoun": ["WP"],
            "wh_adverb": ["WRB"],
        }

    def get_expected_pos_from_corpus(
        self, expected_pos, previous_word, pos_corpus, pos_mapping, p_n_words
    ):
        expected_pos = expected_pos.lower()
        tentative_next_words = self.w2v.word_sim(previous_word, 11)[1:]
        t_pos = nltk.pos_tag(tentative_next_words)
        # print(expected_pos)
        pos_list = pos_mapping[expected_pos]
        tentative_word = ""
        found_pos = False
        for pos in t_pos:
            if pos[1] in pos_list:
                tentative_word = pos[0]
                if tentative_word not in p_n_words:
                    found_pos = True
                    if len(p_n_words) == 5:
                        p_n_words.pop(0)
                    p_n_words.append(tentative_word)
                    break
                else:
                    continue

        if not found_pos:
            try:
                random_expected_pos = np.random.choice(pos_list)
                # print(random_expected_pos)
                word = np.random.choice(pos_corpus[random_expected_pos]).lower()
                if len(p_n_words) == 5:
                    p_n_words.pop(0)
                p_n_words.append(word)
                return word
            except Exception as e:
                if len(p_n_words) == 5:
                    p_n_words.pop(0)
                word = np.random.choice(tentative_next_words)
                p_n_words.append(word)
                return word

        else:
            return tentative_word

    def generate_lyrics_withouttemplate(self, initial_token):
        n_words = 100
        previous_word = initial_token.lower()
        predicted_lyrics = previous_word
        previous_five_words = [previous_word]

        for _ in tqdm(range(n_words)):
            count = 0
            tentaitve_next_words = self.w2v.word_sim(previous_word, 11)[1:]

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

    def get_pos_tags(self):
        sad_song_pos = {}
        pos_tags = nltk.pos_tag(set(self.w2v.words_list))
        for pos in pos_tags:
            if pos[1] in sad_song_pos:
                sad_song_pos[pos[1]].append(pos[0])
            else:
                sad_song_pos[pos[1]] = [pos[0]]
        return pos_tags

    def generate_lyrics_from_template(self, starting_word):
        # template parse verse1, chorus1, bridge, outro
        song_pos = self.get_pos_tags()
        previous_word = starting_word.lower()
        gen_lyric = ""
        previous_n_words = []
        for song_section, structure in self.template.items():
            pos_list = list(map(lambda w: w.strip(), structure.split(",")))
            # print(pos_list)
            for expected_pos in pos_list:
                if (
                    expected_pos not in ["<br>", "<comma>"]
                    and not expected_pos.isspace()
                ):
                    next_word = self.get_expected_pos_from_corpus(
                        expected_pos,
                        previous_word=previous_word,
                        pos_corpus=song_pos,
                        pos_mapping=self.pos_mapping,
                        p_n_words=previous_n_words,
                    )
                    gen_lyric += f"{next_word} "
                    previous_word = next_word
                else:
                    if expected_pos == "<comma>":
                        gen_lyric += ","
                    else:
                        gen_lyric += "\n"
            gen_lyric += "\n\n"

        return gen_lyric
