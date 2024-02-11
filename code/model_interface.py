from cProfile import label
import gradio as gr
import numpy as np
from sentiment_classifier.scripts.embeddings_classifier import EMBClassifier
from sentiment_classifier.scripts.kmeans_clustering import KMeans
from lyrics_generator.lyrics_gen_template import LyricsGenerator
from sentiment_classifier.scripts.nn_classifier import NN
from sentiment_classifier.scripts.neural_network import NeuralNetwork
from lyrics_generator.word2vector import word2vec

wv_from_bin = None


def load_embedding_model():
    """Load Word2vec Embeddings
    Return:
        wv_from_bin: All 3000000 embeddings, each lengh 300
    """
    import gensim.downloader as api

    wv_from_bin = api.load("word2vec-google-news-300")
    print("Loaded vocab size %i" % len(list(wv_from_bin.index_to_key)))
    return wv_from_bin


def classify(lyrics: str, model_selection: str) -> list[str]:
    global wv_from_bin
    if wv_from_bin is None:
        raise Exception("Word embeddings are not assigined, please load the word embeddings")
    
    if model_selection == "Embeddings":
        embeddings_classifier = EMBClassifier()
        return list(
            embeddings_classifier.predict(input_lyrics=lyrics, wv_from_bin=wv_from_bin)
        )
    elif model_selection == "Kmeans":
        kmeans_clustering = KMeans()
        return kmeans_clustering.predict(input_lyrics=lyrics, wv_from_bin=wv_from_bin)

    elif model_selection == "Neural Net":
        nn = NN()
        return list(nn.predict(input_lyrics=lyrics, wv_from_bin=wv_from_bin))
    else:
        raise Exception(f"Not supported {model_selection}")


def choose_mode(
    mode: str, choice: str, lyrics_input: str, emotion: str, starting_word: str
):
    print(f"{mode=}, {choice=}, {emotion=}, {starting_word=}")
    if mode == "Lyrics Generator":
        print(f"{emotion=}, {starting_word=}")
        gen = LyricsGenerator(emotion)
        if choice == "Without Template":
            return gen.generate_lyrics_withouttemplate(starting_word)
        elif choice == "With Template":
            return gen.generate_lyrics_from_template(starting_word)

    elif mode == "Lyrics Classifier":
        return classify(lyrics_input, choice)


if __name__ == "__main__":
    print(f"Loading Word2vec vectors, wait for few seconds")
    wv_from_bin = load_embedding_model()

    options_1 = ["Lyrics Classifier", "Lyrics Generator"]
    options_2 = {
        "Lyrics Classifier": ["Embeddings", "Kmeans", "Neural Net"],
        "Lyrics Generator": ["Without Template", "With Template"],
    }
    emotion_choices = [
        "Joy",
        "Trust",
        "Fear",
        "Surprise",
        "Sadness",
        "Disgust",
        "Anger",
        "Anticipation",
    ]

    with gr.Blocks() as demo:
        mode_type = gr.Dropdown(choices=options_1, label="Modes")
        model_type = gr.Dropdown([], label="Choices")

        def update_second(first_val):
            d2 = gr.Dropdown(options_2[first_val])
            return d2

        mode_type.input(update_second, mode_type, model_type)
        lyrics_input = gr.Textbox(label="Input Lyrics (For lyrics classification)")
        starting_word_input = gr.Textbox(label="Starting word (For lyrics generation)")
        emotion = gr.Dropdown(
            choices=emotion_choices, label="Select motion (For lyrics generation)"
        )
        outputs = gr.Textbox(label="Predictions")
        submit_button = gr.Button(value="Submit", variant="primary")
        submit_button.click(
            choose_mode,
            inputs=[mode_type, model_type, lyrics_input, emotion, starting_word_input],
            outputs=outputs,
        )
    demo.launch(share=True)
