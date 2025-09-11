import pickle
import pandas as pd
import numpy as np
import gensim.downloader as api
from sklearn.model_selection import train_test_split, GridSearchCV
import nltk
nltk.download('averaged_perceptron_tagger_eng')
import nlpaug.augmenter.word as naw
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, make_scorer

df = pd.read_csv('data/synthetic_panic_meter_dataset.csv')

# Getting insights about the data.
print(df.head())
print("================================================================================")
print(df.isnull().sum())
print("================================================================================")
print(df.describe())
print("================================================================================")

def augment_data(input_df, target_count = 15000):
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

    augmenter = naw.SynonymAug(aug_src = 'wordnet')
    augmented_texts, augmented_scores = [], []

    for _, row in input_df.iterrows():
        original_text = row['conversation']
        if not isinstance(original_text, str) or not original_text.strip(): continue
        augmented_batch = augmenter.augment(original_text, n = aug_per_text)
        if not isinstance(augmented_batch, list): augmented_batch = [augmented_batch]
        for text in augmented_batch:
            augmented_texts.append(text)
            augmented_scores.append(row['panic_meter'])

    augmented_df = pd.DataFrame({'conversation': augmented_texts, 'panic_meter': augmented_scores})
    final_df = pd.concat([input_df, augmented_df], ignore_index=True)
    final_df.to_csv("Augmented_data.csv", index = False)
    return final_df.sample(n=target_count, random_state=101).reset_index(drop=True)

augment_data(input_df = df, target_count = 20000)

aug_df = pd.read_csv("data/Augmented_data.csv")
fasttext = api.load("glove-twitter-25")  # Using a smaller model to handle during deployment.

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

aug_df["vector_embeddings"] = aug_df["conversation"].apply(lambda x: sentence_embeddings(x))

embeddings = np.vstack(aug_df["vector_embeddings"].values)

# Creating separate columns for each vector element
embedding_df = pd.DataFrame(embeddings,
                            columns = [f"emb_{i}" for i in range(embeddings.shape[1])])
preprocessed_df = pd.concat([aug_df.drop(columns = ["vector_embeddings"]), embedding_df], axis = 1)

preprocessed_df.to_csv("Embedded_dataset.csv", index = False)

fasttext.save("glove_model.kv")  # Saving the fasttext model.

embedded_df = pd.read_csv("data/Embedded_dataset.csv")

# Splitting the dataset
X = embedded_df.drop(["panic_meter", "conversation"], axis = 1, inplace = False)
y = embedded_df["panic_meter"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 21 )

# Training the model
xgb_model = XGBRegressor(n_jobs = -1, reg_lambda = 1)   # reg_lambda penalizes the mse and also called as ridge. It decreases the impact of all features.
# Using l2(reg_lambda) instead of l1(reg_alpha) because l1 is mostly used to eliminate the features which are unnecessary by forcing them to be zero.
linear_regressor_model = LinearRegression(n_jobs = -1)
lgbm_model = LGBMRegressor(n_jobs = -1)
rf_model = RandomForestRegressor(n_jobs = -1)

estimators = [("xgb", xgb_model),
              ("linear_regression", linear_regressor_model),
              ("lgbm", lgbm_model)]

model = StackingRegressor(estimators = estimators, final_estimator = rf_model, n_jobs = 4, verbose = 4)
model.fit(X_train, y_train)
with open("models/stacking_model.pkl", mode = "wb+") as f:
    pickle.dump(model, f)

# Testing the model
train_pred = model.predict(X_train)
train_mse = mean_squared_error(y_train, train_pred)
train_mae = mean_absolute_error(y_train, train_pred)
print(f"Train mean squarred error of {str(model.__class__.__name__)} : {train_mse} | Train mean absolute error of {str(model.__class__.__name__)} : {train_mae}")

test_pred = model.predict(X_test)
test_mse = mean_squared_error(y_test, test_pred)
test_mae = mean_absolute_error(y_test, test_pred)
print(f"Test mean squarred error of {str(model.__class__.__name__)} : {test_mse} | Test mean absolute error of {str(model.__class__.__name__)} : {test_mae}")
