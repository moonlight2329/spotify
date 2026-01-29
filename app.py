# app.py
# NLP-Based Spotify Recommendation System
# Dataset: data/top_100_spotify_songs_2025.csv

import streamlit as st
import pandas as pd
import re
import os
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

# -----------------------------
# Text preprocessing (NLP)
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
st.write("Genre-based music recommendation using NLP and popularity ranking.")

# Load data
data = load_data()

# Ensure Popularity_Score is numeric
if 'Popularity_Score' in data.columns:
    data['Popularity_Score'] = pd.to_numeric(
        data['Popularity_Score'],
        errors='coerce'
    ).fillna(0)

st.subheader("Dataset Preview")
st.dataframe(data.head())

# -----------------------------
# Column validation
# -----------------------------
required_columns = ['Song_Title', 'Artist', 'Genre', 'Popularity_Score']

missing_cols = [col for col in required_columns if col not in data.columns]
if missing_cols:
    st.error(f"❌ Missing columns: {missing_cols}")
    st.stop()

# -----------------------------
# NLP text preparation (optional but academic)
# -----------------------------
data['combined_text'] = (
    data['Song_Title'].astype(str) + " " +
    data['Artist'].astype(str) + " " +
    data['Genre'].astype(str)
)

data['clean_text'] = data['combined_text'].apply(preprocess_text)

@st.cache_data
def build_tfidf(corpus):
    vectorizer = TfidfVectorizer(max_features=3000)
    matrix = vectorizer.fit_transform(corpus)
    return vectorizer, matrix

vectorizer, tfidf_matrix = build_tfidf(data['clean_text'])

# -----------------------------
# User Input
# -----------------------------
st.subheader("🎼 Choose Your Genre")
user_genre = st.text_input("Enter a genre (e.g. pop, rock, hip hop)")

# -----------------------------
# Recommendation Logic
# -----------------------------
if st.button("Recommend Songs"):
    if user_genre.strip() == "":
        st.warning("Please enter a genre")
    else:
        genre = user_genre.strip().lower()

        # Filter by genre
        filtered = data[data['Genre'].str.lower() == genre]

        if filtered.empty:
            st.warning("No songs found for this genre")
        else:
            # Find top artists in this genre
            top_artists = (
                filtered['Artist']
                .value_counts()
                .head(3)
                .index
            )

            # Get recommendations
            recommendations = filtered[
                filtered['Artist'].isin(top_artists)
            ].copy()

            # Sort by popularity score
            recommendations = recommendations.sort_values(
                by='Popularity_Score',
                ascending=False
            ).head(5)

            # -----------------------------
            # Display Recommendations
            # -----------------------------
            st.subheader(f"🎧 Top {genre.title()} Recommendations")

            for _, row in recommendations.iterrows():
                st.write(f"🎵 **{row['Song_Title']}**")
                st.caption(
                    f"Artist: {row['Artist']} | "
                    f"Popularity Score: {row['Popularity_Score']}"
                )

            # -----------------------------
            # 📊 Popularity-Based Graph
            # -----------------------------
            st.subheader("📊 Recommendation Analysis (Popularity Score)")

            fig, ax = plt.subplots()
            ax.barh(
                recommendations['Song_Title'],
                recommendations['Popularity_Score']
            )
            ax.set_xlabel("Popularity Score")
            ax.set_ylabel("Recommended Songs")
            ax.invert_yaxis()

            st.pyplot(fig)
