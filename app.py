# app.py
# NLP-Based Spotify Music Recommendation System using TF-IDF

import streamlit as st
import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


# Uncomment and run once locally if needed
# nltk.download('stopwords')
# nltk.download('wordnet')

# -----------------------------
# Text preprocessing setup
# -----------------------------

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    tokens = text.split()
    tokens = [w for w in tokens if w not in ENGLISH_STOP_WORDS]
    return ' '.join(tokens)


# -----------------------------
# Load dataset
# -----------------------------
@st.cache_data
def load_data():
    # IMPORTANT: adjust column names if needed
    df = pd.read_csv('data/spotify_data_clean.csv')


    # Example assumption:
    # 'track_name' -> song title
    # 'artist_name' -> artist
    # 'lyrics' or 'track_genre' -> text feature

    text_column = 'track_genre'  # CHANGE if your dataset uses a different column

    df['clean_text'] = df[text_column].apply(preprocess_text)
    return df

# -----------------------------
# Build TF-IDF model
# -----------------------------
@st.cache_data
def build_tfidf(corpus):
    vectorizer = TfidfVectorizer(max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(corpus)
    return vectorizer, tfidf_matrix

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🎵 Spotify Music Recommendation System")
st.write("Recommend songs based on NLP similarity using Spotify metadata.")

# Load data
with st.spinner("Loading Spotify dataset..."):
    data = load_data()

st.subheader("Dataset Preview")
st.dataframe(data.head())

# Build model
vectorizer, tfidf_matrix = build_tfidf(data['clean_text'])

# User input
st.subheader("Describe your music preference")
user_input = st.text_area("Example: chill electronic music with soft vocals")

if st.button("Recommend Songs"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        clean_input = preprocess_text(user_input)
        user_vec = vectorizer.transform([clean_input])
        similarity = cosine_similarity(user_vec, tfidf_matrix)
        top_indices = similarity.argsort()[0][-5:][::-1]

        st.subheader("Recommended Tracks")
        for idx in top_indices:
            st.write(f"**{data.iloc[idx]['track_name']}**")
            if 'artist_name' in data.columns:
                st.caption(f"Artist: {data.iloc[idx]['artist_name']}")
