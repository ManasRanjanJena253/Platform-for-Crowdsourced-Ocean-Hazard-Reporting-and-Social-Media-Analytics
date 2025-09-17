import pickle
import gensim.downloader as api
import numpy as np
import time

with open("models/randomforest_model.pkl", mode="rb+") as f:
    panic_meter_model = pickle.load(f)
    print("model_loaded")

fasttext = api.load("glove-twitter-25")
def sentence_embeddings(text):
    """
    Used for preprocessing the text and converting them into vector embeddings using Glove-twitter model.
    :param text: The text to be embedded.
    :return: vector embeddings.
    """
    # The model is already trained on twitter data, so not removing the emojis as they may carry important emotions, and this won't cause any error.
    words = text.lower().split()
    vectors = [fasttext[w] for w in words if w in fasttext]
    if len(vectors) == 0:
        return np.zeros(25)
    return np.mean(vectors, axis = 0)

try:
    start_time = time.perf_counter()
    tweet = "Power cut in whole area due to cyclone, scary night"
    embeddings = sentence_embeddings(tweet)
    end_time = time.perf_counter()
    print("Perf counter Time taken : ", end_time - start_time)
    print("Embeddings Created")

    panic_meter = panic_meter_model.predict(embeddings.reshape(1, -1))
    print(panic_meter)
except Exception as e:
    print(str(e))
