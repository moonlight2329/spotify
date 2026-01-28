# app.py
# NLP-Based Spotify Recommendation System
# Dataset: data/Popular_Spotify_Songs.csv

import streamlit as st
import pandas as pd
import re
import os

from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Text preprocessing
# -----------------------------
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    tokens = text.split()
    tokens = [w for w in tokens if w not in ENGLISH_STOP_WORDS]
    return ' '.join(tokens)

# -----------------------------
# Load dataset safely
# -----------------------------
@st.cache_data
def load_data():
    file_path = 'data/top_100_spotify_songs_2025.csv'

    if not os.path.exists(file_path):
        st.error(f"❌ Dataset not found at {file_path}")
        st.stop()

    df = pd.read_csv(file_path)
    return df

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🎵 Spotify Music Recommendation System")
st.write("Personalized music recommendations using NLP and TF-IDF.")

# Load data
data = load_data()

st.subheader("Dataset Preview")
st.dataframe(data.head())

# -----------------------------
# Select text column
# -----------------------------
# CHANGE this if your dataset uses a different column
text_column = 'Genre'

if text_column not in data.columns:
    st.error(f"❌ Column '{text_column}' not found in dataset")
    st.write("Available columns:", data.columns.tolist())
    st.stop()

# Clean text
data['combined_text'] = (
    data['Song_Title'].astype(str) + " " +
    data['Artist'].astype(str) + " " +
    data['Genre'].astype(str)

)

data['clean_text'] = data['combined_text'].apply(preprocess_text)


# -----------------------------
# TF-IDF model
# -----------------------------
@st.cache_data
def build_tfidf(corpus):
    vectorizer = TfidfVectorizer(max_features=5000)
    matrix = vectorizer.fit_transform(corpus)
    return vectorizer, matrix

vectorizer, tfidf_matrix = build_tfidf(data['clean_text'])

# -----------------------------
# User input
# -----------------------------
st.subheader("Describe your music taste")
user_input = st.text_area("Example: energetic pop dance music")

if st.button("Recommend Songs"):
    if user_input.strip() == "":
        st.warning("Please enter some text")
    else:
        clean_input = preprocess_text(user_input)
        user_vec = vectorizer.transform([clean_input])
        similarity = cosine_similarity(user_vec, tfidf_matrix)[0]

        top_indices = similarity.argsort()[-5:][::-1]

        st.subheader("🎧 Recommended Tracks")
        for idx in top_indices:
        track = data.iloc[idx]
        st.write(f"🎵 **{track['Song_Title']}**")
        st.caption(f"Artist: {track['Artist']}")
