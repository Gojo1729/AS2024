# 2024

CodeAThon 2024 Feb

# Demo

Checkout the live demo at - https://huggingface.co/spaces/Shiv1729/AS-2024-Demo

# How to testout the models?

1. cd into code directory
1. Install all the packages in `pip install -f requirements.txt`
1. Launch Gradio using `python model_interface.py`

# Folder strucutre

- All the code files are in the code/ directory
- lyrics_generator has the code related to generating the lyrics
- sentiment_classifier has the files related to classifying the song lyrics.
- Each folder has scripts, notebooks, embeddings subfolder.
  - scripts - python scripts belonging to each algorithm.
  - notebooks - notebooks which were used to train the models.
  - embeddings - saved pickle state of the model objects, these pickle files are used while inferencing the model in Gradio.

# Predictions

- Find the predicted label for Spotify dataset inside code/sentiment_classifier/results
- spotify_classification_kmeans.csv - KMeans predictions
- spotify_classification_nn.csv - Neural network predictions

# Tensorboard

- You can checkout the experiment logs by launching tensorboard session inside code/sentiment_classifier and code/lyrics_generator

# Libraries used.

NumPy – For implementing the algorithms.
Pandas – Data processing
Gensim – For loading the pre-trained word2vec algorithms.
NLTK – For processing the text (tokenization, lemmatization, POS tagging)
Tensorboard – For experiment tracking

# Note:

- As mentioned before for faster inference the models are saved in pickle file state and to be reused in inferencing.
- When you first run the "python code/model_interface.py", I download the word2vec pre-trained embeddings, this can be time consuming as it needs
  to download the files for the first time. This may take anywhere between 3-4 minutes in the first run, subsequently it would take 45 seconds to launch the Gradio
  interface.
