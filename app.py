import streamlit as st
import pandas as pd
from recommendation import recommend_content, recommend_collaborative, hybrid_recommendation

# App Title
st.title("🎬 Netflix Recommendation System")

# Sidebar for Recommendation Type
option = st.sidebar.selectbox(
    "Choose Recommendation Type:",
    ("Content-Based", "Collaborative", "Hybrid")
)

# Load data to populate dropdowns
movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

# Input fields
user_ids = ratings['userId'].unique()
movie_titles = movies['title'].sort_values().tolist()

if option == "Content-Based":
    selected_movie = st.selectbox("Select a Movie:", movie_titles)
    if st.button("Get Recommendations"):
        recs = recommend_content(selected_movie)
        st.write("Recommended Movies:")
        for r in recs:
            st.write(f"• {r}")

elif option == "Collaborative":
    selected_user = st.selectbox("Select a User ID:", sorted(user_ids))
    if st.button("Get Recommendations"):
        recs = recommend_collaborative(selected_user)
        st.write("Recommended Movies:")
        for r in recs:
            st.write(f"• {r}")

else:  # Hybrid
    selected_user = st.selectbox("Select a User ID:", sorted(user_ids))
    selected_movie = st.selectbox("Select a Movie:", movie_titles)
    if st.button("Get Recommendations"):
        recs = hybrid_recommendation(selected_user, selected_movie)
        st.write("Hybrid Recommendations:")
        for r in recs:
            st.write(f"• {r}")
