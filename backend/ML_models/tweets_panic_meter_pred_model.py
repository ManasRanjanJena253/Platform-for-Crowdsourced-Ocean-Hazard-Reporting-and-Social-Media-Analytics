import pickle

import nltk
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
nltk.download('averaged_perceptron_tagger_eng')
import nlpaug.augmenter.word as naw
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

df = pd.read_csv('data/synthetic_panic_meter_dataset.csv')

# Getting insights about the data.
print(df.head())
print("================================================================================")
print(df.isnull().sum())
print("================================================================================")
print(df.describe())
print("================================================================================")

def augment_data(input_df, target_count=15000):
    """
    Used for augmenting the given dataset using nlpaug to increase its size for training.
    :param input_df: The original dataset.
    :param target_count: The no. of rows the new augmented dataset should have.
    :return:
    """
    if len(input_df) >= target_count:
        return input_df
    print(f"\nAugmenting data from {len(input_df)} to {target_count} rows...")
    num_to_generate = target_count - len(input_df)
    aug_per_text = int(np.ceil(num_to_generate / len(input_df)))

    augmenter = naw.SynonymAug(aug_src='wordnet')
    augmented_texts, augmented_scores = [], []

    for _, row in input_df.iterrows():
        original_text = row['conversation']
        if not isinstance(original_text, str) or not original_text.strip():
            continue
        augmented_batch = augmenter.augment(original_text, n=aug_per_text)
        if not isinstance(augmented_batch, list):
            augmented_batch = [augmented_batch]
        for text in augmented_batch:
            augmented_texts.append(text)
            augmented_scores.append(row['panic_meter'])

    augmented_df = pd.DataFrame({'conversation': augmented_texts, 'panic_meter': augmented_scores})
    final_df = pd.concat([input_df, augmented_df], ignore_index=True)
    final_df.to_csv("data/Augmented_data.csv", index=False)
    return final_df.sample(n=target_count, random_state=101).reset_index(drop=True)

# Augmenting data
augment_data(input_df = df)

# Load augmented dataset
aug_df = pd.read_csv("data/Augmented_data.csv")

# Vectorization with TF-IDF + Truncated SVD
vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=10000)
X_tfidf = vectorizer.fit_transform(aug_df["conversation"])
with open("models/tf_idf_vectorizer.pkl", mode = "wb+") as f:
    pickle.dump(vectorizer, f)

# Dimensionality reduction
svd = TruncatedSVD(n_components=300, random_state=42)
X_reduced = svd.fit_transform(X_tfidf)
with open("models/svd.pkl", mode = "wb+") as f:
    pickle.dump(svd, f)

# Final preprocessed dataframe
embedding_df = pd.DataFrame(X_reduced, columns=[f"emb_{i}" for i in range(X_reduced.shape[1])])
preprocessed_df = pd.concat([aug_df.reset_index(drop=True), embedding_df], axis=1)
preprocessed_df.to_csv("data/Embedded_dataset.csv", index=False)
print("Embedding generated successfully")

# Load processed dataset
embedded_df = pd.read_csv("data/Embedded_dataset_panic_level.csv")

# Splitting the dataset
X = embedded_df.drop(["panic_urgency", "conversation"], axis=1, inplace=False)
y = embedded_df["panic_urgency"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

# Training the models
model = LGBMClassifier(n_jobs = 4)

model.fit(X_train, y_train)

# Save the model
with open("models/lgbm_classifier_model.pkl", mode = "wb+") as f:
    pickle.dump(model, f)

# Testing the model
train_pred = model.predict(X_train)
print("Train Classification report : ", classification_report(y_train, train_pred))

test_pred = model.predict(X_test)
print("Test Classification report : ", classification_report(y_test, test_pred))
