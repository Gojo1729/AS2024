import gradio as gr
import numpy as np
from sentiment_classifier.embeddings_classifier import EMBClassifier

wv_from_bin = None


def predict(lyrics: str, model_selection: list[str]) -> list[str]:
    embeddings_classifier = EMBClassifier()
    global wv_from_bin

    return list(
        embeddings_classifier.predict(input_lyrics=lyrics, wv_from_bin=wv_from_bin)
    )


def load_embedding_model():
    """Load GloVe Vectors
    Return:
        wv_from_bin: All 400000 embeddings, each lengh 200
    """
    import gensim.downloader as api

    wv_from_bin = api.load("word2vec-google-news-300")
    print("Loaded vocab size %i" % len(list(wv_from_bin.index_to_key)))
    return wv_from_bin


if __name__ == "__main__":
    print(f"Loading Glove vectors, wait for few minutes")
    wv_from_bin = load_embedding_model()
    # Create Gradio interface with dropdown menu
    inputs = gr.Textbox(label="Input Lyrics")
    model_dropdown = gr.Dropdown(
        choices=["Embeddings approach", "KMeans", "Logistic Regression"],
        label="Select Model",
    )
    outputs = gr.Textbox(label="Sentiment/Mood")
    interface = gr.Interface(
        fn=predict, inputs=[inputs, model_dropdown], outputs=outputs
    )
    interface.launch()
