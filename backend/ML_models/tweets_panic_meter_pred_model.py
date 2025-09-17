import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import nltk
nltk.download('averaged_perceptron_tagger_eng')
import nlpaug.augmenter.word as naw
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
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

# Dimensionality reduction
svd = TruncatedSVD(n_components=300, random_state=42)
X_reduced = svd.fit_transform(X_tfidf)

# Final preprocessed dataframe
embedding_df = pd.DataFrame(X_reduced, columns=[f"emb_{i}" for i in range(X_reduced.shape[1])])
preprocessed_df = pd.concat([aug_df.reset_index(drop=True), embedding_df], axis=1)
preprocessed_df.to_csv("data/Embedded_dataset.csv", index=False)
print("Embedding generated successfully")

# Load processed dataset
embedded_df = pd.read_csv("data/Embedded_dataset.csv")

# Splitting the dataset
X = embedded_df.drop(["panic_meter", "conversation"], axis=1, inplace=False)
y = embedded_df["panic_meter"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

# Training the models
xgb_model = XGBRegressor(n_jobs = -1, reg_lambda = 2)
linear_regressor_model = LinearRegression(n_jobs = -1)
lgbm_model = LGBMRegressor(n_jobs = -1)
rf_model = RandomForestRegressor(n_jobs = -1)
svr = SVR()
ridge = Ridge()
mlp_reg = MLPRegressor(random_state = 21)

estimators = [
    ("svr", svr),
    ("xgb", xgb_model),
    ("ridge", ridge)
]

model = StackingRegressor(estimators = estimators, final_estimator = mlp_reg, n_jobs = 4, verbose = 10, cv = 5)

model.fit(X_train, y_train)

# Save the model
with open("models/stacking_regressor(mpl_reg_final_estimator)_model.pkl", mode = "wb+") as f:
    pickle.dump(model, f)

# Testing the model
train_pred = model.predict(X_train)
train_mse = mean_squared_error(y_train, train_pred)
train_mae = mean_absolute_error(y_train, train_pred)
print(f"Train mean squared error of {str(model.__class__.__name__)} : {train_mse} | Train mean absolute error : {train_mae}")

test_pred = model.predict(X_test)
test_mse = mean_squared_error(y_test, test_pred)
test_mae = mean_absolute_error(y_test, test_pred)
print(f"Test mean squared error of {str(model.__class__.__name__)} : {test_mse} | Test mean absolute error : {test_mae}")
