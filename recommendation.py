import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
import numpy as np

# -----------------------
# Load Data
# -----------------------
movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

# Preprocessing for genres
movies['genres'] = movies['genres'].fillna('')

# Create movieId-to-title and title-to-index mappings
movie_id_to_title = dict(zip(movies['movieId'], movies['title']))
title_to_index = pd.Series(movies.index, index=movies['title']).drop_duplicates()

# -----------------------
# Content-Based Filtering
# -----------------------
def build_content_model():
    tfidf = TfidfVectorizer(token_pattern=r'[^|]+')
    tfidf_matrix = tfidf.fit_transform(movies['genres'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    return cosine_sim

cosine_sim = build_content_model()

def recommend_content(title, top_n=5):
    if title not in title_to_index:
        return ["Movie not found."]
    idx = title_to_index[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices].tolist()

# -----------------------
# Collaborative Filtering (User-Based)
# -----------------------
def build_collaborative_model():
    user_movie_ratings = ratings.pivot_table(index='userId', columns='movieId', values='rating')
    user_movie_ratings.fillna(0, inplace=True)
    user_similarity = cosine_similarity(user_movie_ratings)
    return user_movie_ratings, pd.DataFrame(user_similarity, index=user_movie_ratings.index, columns=user_movie_ratings.index)

user_movie_ratings, user_similarity_df = build_collaborative_model()

def recommend_collaborative(user_id, top_n=5):
    if user_id not in user_similarity_df.index:
        return ["User not found."]
    similar_users = user_similarity_df[user_id].sort_values(ascending=False)[1:]
    top_users = similar_users.head(5).index
    top_ratings = user_movie_ratings.loc[top_users].mean().sort_values(ascending=False)

    # Exclude movies already rated by the user
    watched = user_movie_ratings.loc[user_id][user_movie_ratings.loc[user_id] > 0].index
    recommendations = top_ratings.drop(watched, errors='ignore').head(top_n)
    return [movie_id_to_title[mid] for mid in recommendations.index if mid in movie_id_to_title]

# -----------------------
# Hybrid Recommendation
# -----------------------
def hybrid_recommendation(user_id, title, top_n=5):
    content_recs = recommend_content(title, top_n * 2)
    collab_recs = recommend_collaborative(user_id, top_n * 2)
    combined = pd.Series(content_recs + collab_recs)
    top_recs = combined.value_counts().head(top_n).index.tolist()
    return top_recs

# -----------------------
# Sample Calls
# -----------------------
if __name__ == "__main__":
    print("Content-Based Recommendations for 'Toy Story':")
    print(recommend_content('Toy Story'))

    print("\n Collaborative Recommendations for User 1:")
    print(recommend_collaborative(1))

    print("\n Hybrid Recommendations for User 1 and 'Toy Story':")
    print(hybrid_recommendation(1, 'Toy Story'))

