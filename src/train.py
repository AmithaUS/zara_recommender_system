import pandas as pd
import numpy as np
import pickle
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# -----------------------
# Load Dataset
# -----------------------

df = pd.read_csv("data/zara_products.csv", sep=";")

# -----------------------
# Cleaning
# -----------------------

df = df.drop_duplicates()
df = df.dropna(subset=["name", "description", "price"])

df["price"] = pd.to_numeric(df["price"], errors="coerce")
df = df.dropna(subset=["price"])

df = df.reset_index(drop=True)

# -----------------------
# Feature Engineering
# -----------------------

df["combined_features"] = (
    df["name"].astype(str) + " " +
    df["description"].astype(str) + " " +
    df["Product Category"].astype(str) + " " +
    df["section"].astype(str)
)

# Normalize price
scaler = MinMaxScaler()
df["normalized_price"] = scaler.fit_transform(df[["price"]])

# -----------------------
# Train-Test Split
# -----------------------

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42
)

train_df = train_df.reset_index(drop=True)

# -----------------------
# TF-IDF
# -----------------------

tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=6000
)

X_train_text = tfidf.fit_transform(train_df["combined_features"])

# -----------------------
# Similarity
# -----------------------

text_similarity = cosine_similarity(X_train_text)

price_similarity = cosine_similarity(
    train_df[["normalized_price"]]
)

hybrid_similarity = (0.8 * text_similarity) + (0.2 * price_similarity)

# -----------------------
# Save Everything
# -----------------------

os.makedirs("models", exist_ok=True)

pickle.dump(tfidf, open("models/tfidf.pkl", "wb"))
pickle.dump(hybrid_similarity, open("models/similarity.pkl", "wb"))
pickle.dump(train_df, open("models/train_df.pkl", "wb"))

print("✅ Training Complete. Model Saved.")
