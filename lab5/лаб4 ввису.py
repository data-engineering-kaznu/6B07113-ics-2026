import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# =========================
# 1. Загрузка данных
# =========================
movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

print("Movies:")
print(movies.head())
print("\nRatings:")
print(ratings.head())

# =========================
# 2. Content-Based Filtering
# =========================
movies = movies.dropna(subset=["genres"])
movies["genres"] = movies["genres"].astype(str)

vectorizer = CountVectorizer(token_pattern=r'[^|]+')
genre_matrix = vectorizer.fit_transform(movies["genres"])
movie_similarity = cosine_similarity(genre_matrix, genre_matrix)

movie_indices = pd.Series(movies.index, index=movies["title"]).drop_duplicates()

def recommend_content(movie_title, top_n=5):
    if movie_title not in movie_indices:
        return pd.DataFrame()

    idx = movie_indices[movie_title]
    sim_scores = list(enumerate(movie_similarity[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]

    movie_ids = [i[0] for i in sim_scores]
    scores = [i[1] for i in sim_scores]

    result = movies.iloc[movie_ids][["title", "genres"]].copy()
    result["similarity"] = scores
    return result

print("\nContent-based recommendations for Toy Story (1995):")
content_result = recommend_content("Toy Story (1995)", 5)
print(content_result)

# =========================
# 3. Collaborative Filtering
# =========================
user_movie_matrix = ratings.pivot_table(
    index="userId",
    columns="movieId",
    values="rating"
).fillna(0)

user_similarity = cosine_similarity(user_movie_matrix)
user_similarity_df = pd.DataFrame(
    user_similarity,
    index=user_movie_matrix.index,
    columns=user_movie_matrix.index
)

def recommend_collaborative(user_id, top_n=5):
    if user_id not in user_movie_matrix.index:
        return pd.DataFrame()

    similar_users = user_similarity_df[user_id].sort_values(ascending=False)[1:11]
    watched_movies = user_movie_matrix.loc[user_id]
    watched_movies = watched_movies[watched_movies > 0].index.tolist()

    predicted_scores = {}

    for movie_id in user_movie_matrix.columns:
        if movie_id not in watched_movies:
            ratings_by_similar = user_movie_matrix.loc[similar_users.index, movie_id]
            score = np.dot(similar_users.values, ratings_by_similar.values)

            if similar_users.sum() != 0:
                predicted_scores[movie_id] = score / similar_users.sum()

    top_movies = sorted(predicted_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

    result = pd.DataFrame(top_movies, columns=["movieId", "predicted_rating"])
    result = result.merge(movies[["movieId", "title", "genres"]], on="movieId")
    return result[["movieId", "title", "genres", "predicted_rating"]]

example_user = 1
print(f"\nCollaborative recommendations for user {example_user}:")
collab_result = recommend_collaborative(example_user, 5)
print(collab_result)

# =========================
# 4. Оценка качества: RMSE
# =========================
train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)

train_matrix = train_data.pivot_table(
    index="userId",
    columns="movieId",
    values="rating"
).fillna(0)

test_matrix = test_data.pivot_table(
    index="userId",
    columns="movieId",
    values="rating"
).fillna(0)

common_users = train_matrix.index.intersection(test_matrix.index)
common_movies = train_matrix.columns.intersection(test_matrix.columns)

train_matrix = train_matrix.loc[common_users, common_movies]
test_matrix = test_matrix.loc[common_users, common_movies]

train_similarity = cosine_similarity(train_matrix)
train_similarity_df = pd.DataFrame(
    train_similarity,
    index=train_matrix.index,
    columns=train_matrix.index
)

y_true = []
y_pred = []

for user_id in test_matrix.index:
    similar_users = train_similarity_df[user_id].sort_values(ascending=False)[1:11]

    for movie_id in test_matrix.columns:
        true_rating = test_matrix.loc[user_id, movie_id]

        if true_rating > 0:
            ratings_by_similar = train_matrix.loc[similar_users.index, movie_id]
            score = np.dot(similar_users.values, ratings_by_similar.values)

            if similar_users.sum() != 0:
                predicted_rating = score / similar_users.sum()
            else:
                predicted_rating = 0

            y_true.append(true_rating)
            y_pred.append(predicted_rating)

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
print("\nRMSE:", round(rmse, 4))

# =========================
# 5. Precision@K
# =========================
def precision_at_k(user_id, k=5, threshold=4.0):
    recs = recommend_collaborative(user_id, k)
    if recs.empty:
        return 0

    relevant = 0
    for _, row in recs.iterrows():
        movie_id = row["movieId"]
        real_rating = ratings[
            (ratings["userId"] == user_id) &
            (ratings["movieId"] == movie_id)
        ]["rating"]

        if not real_rating.empty and real_rating.values[0] >= threshold:
            relevant += 1

    return relevant / k

precision = precision_at_k(example_user, 5)
print("Precision@5:", round(precision, 4))

# =========================
# 6. Визуализация рекомендаций
# =========================
if not collab_result.empty:
    plt.figure(figsize=(10, 5))
    plt.barh(collab_result["title"], collab_result["predicted_rating"])
    plt.xlabel("Predicted rating")
    plt.ylabel("Movies")
    plt.title(f"Recommended movies for user {example_user}")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()