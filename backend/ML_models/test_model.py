import pickle
import gensim.downloader as api
import numpy as np
import time

with open("models/lgbm_classifier_model.pkl", mode="rb+") as f:
    panic_meter_model = pickle.load(f)
    print("model_loaded")

with open("models/svd.pkl", "rb") as f:
    svd = pickle.load(f)

with open("models/tf_idf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

fasttext = api.load("glove-twitter-25")
def sentence_embeddings(text):
    """
    Used for preprocessing the text and converting them into vector embeddings using Glove-twitter model.
    :param text: The text to be embedded.
    :return: vector embeddings.
    """

    vectors = vectorizer.transform([text])
    vectorized_text = svd.transform(vectors)

    return vectorized_text

try:
    start_time = time.perf_counter()
    tweet = "Power cut in whole area due to cyclone, scary night"
    embeddings = sentence_embeddings(tweet)
    end_time = time.perf_counter()
    print("Perf counter Time taken : ", end_time - start_time)
    print("Embeddings Created")

    panic_meter = panic_meter_model.predict(embeddings)
    print(panic_meter)
except Exception as e:
    print(str(e))
