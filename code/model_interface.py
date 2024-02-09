from cProfile import label
import gradio as gr
import numpy as np
from sentiment_classifier.embeddings_classifier import EMBClassifier
from sentiment_classifier.kmeans_clustering import KMeans
from lyrics_generator.lyrics_gen import Generator
from sentiment_classifier.nn_classifier import NN
from sentiment_classifier.neural_network import NeuralNetwork

wv_from_bin = None


def load_embedding_model():
    """Load GloVe Vectors
    Return:
        wv_from_bin: All 3000000 embeddings, each lengh 300
    """
    import gensim.downloader as api

    wv_from_bin = api.load("word2vec-google-news-300")
    print("Loaded vocab size %i" % len(list(wv_from_bin.index_to_key)))
    return wv_from_bin


def classify(lyrics: str, model_selection: str) -> list[str]:
    global wv_from_bin
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
        return nn.predict(input_lyrics=lyrics, wv_from_bin=wv_from_bin)


def choose_mode(mode: str, choice: str, lyrics_input: str):
    if mode == "Lyrics Generator":
        gen = Generator()
        return gen.generate_lyrics(choice)

    elif mode == "Lyrics Classifier":
        return classify(lyrics_input, choice)

        # Create Gradio interface with dropdown menu


if __name__ == "__main__":
    print(f"Loading Glove vectors, wait for few seconds")
    wv_from_bin = load_embedding_model()
    options_1 = ["Lyrics Classifier", "Lyrics Generator"]
    options_2 = {
        "Lyrics Classifier": ["Embeddings", "Kmeans", "Neural Net"],
        "Lyrics Generator": ["Sad", "Joy", "Fear"],
    }

    with gr.Blocks() as demo:
        mode_type = gr.Dropdown(choices=options_1, label="Modes")
        choice = gr.Dropdown([], label="Choices")

        def update_second(first_val):
            d2 = gr.Dropdown(options_2[first_val])
            return d2

        mode_type.input(update_second, mode_type, choice)
        lyrics_input = gr.Textbox(label="Input Lyrics")
        outputs = gr.Textbox("Output")

        # def print_results(option_1, option_2):
        #     return f"You selected '{option_1}' in the first dropdown and '{option_2}' in the second dropdown."

        # choice.input(choose_mode, [mode_type, choice, lyrics_input], outputs)
        gr.Interface
        submit_button = gr.Button(value="Submit", variant="primary")
        submit_button.click(
            choose_mode,
            inputs=[mode_type, choice, lyrics_input],
            outputs=outputs,
        )
    demo.launch(share=True)
