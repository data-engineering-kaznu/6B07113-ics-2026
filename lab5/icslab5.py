import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
ratings = pd.read_csv("input5.csv")
movies = pd.read_csv("input5.1.csv")
data = pd.merge(ratings, movies, on="movieId")
genres = movies['genres'].str.get_dummies('|')
movie_similarity = cosine_similarity(genres)
movie_similarity_df = pd.DataFrame(
    movie_similarity,
    index=movies['movieId'],
    columns=movies['movieId']
)
def recommend_content(movie_id, top_n=5):
    sim_scores = movie_similarity_df[movie_id].sort_values(ascending=False)
    sim_scores = sim_scores.iloc[1:top_n+1]
    return sim_scores.index.tolist()
user_movie_matrix = ratings.pivot_table(
    index='userId',
    columns='movieId',
    values='rating'
)
user_movie_matrix = user_movie_matrix.fillna(0)
user_similarity = cosine_similarity(user_movie_matrix)
user_similarity_df = pd.DataFrame(
    user_similarity,
    index=user_movie_matrix.index,
    columns=user_movie_matrix.index
)
def recommend_user(user_id, top_n=5):
    similar_users = user_similarity_df[user_id].sort_values(ascending=False)
    similar_users = similar_users.iloc[1:6]
    movies_watched = user_movie_matrix.loc[user_id]
    movies_watched = movies_watched[movies_watched > 0].index
    scores = {}
    for sim_user in similar_users.index:
        sim_score = similar_users[sim_user]
        sim_movies = user_movie_matrix.loc[sim_user]
        for movie in sim_movies.index:
            if movie not in movies_watched and sim_movies[movie] > 0:
                if movie not in scores:
                    scores[movie] = 0
                scores[movie] += sim_score * sim_movies[movie]
    recommended = sorted(scores, key=scores.get, reverse=True)[:top_n]
    return recommended
pred = []
true = []
for row in ratings.itertuples():
    user = row.userId
    rating = row.rating
    pred_rating = user_movie_matrix.loc[user].mean()
    pred.append(pred_rating)
    true.append(rating)
rmse = np.sqrt(mean_squared_error(true, pred))
print("RMSE:", rmse)
results = []
users = ratings['userId'].unique()[:20]   
for user in users:
    rec_movies = recommend_user(user)
    for movie in rec_movies:
        movie_row = movies[movies['movieId'] == movie]
        if len(movie_row) > 0:
            title = movie_row['title'].values[0]
        else:
            title = "Unknown"
        results.append({
            "userId": user,
            "movieId": movie,
            "title": title
        })
output = pd.DataFrame(results)
output.to_csv("output5.csv", index=False)
print("Файл output5.csv создан")
top_movies = output['title'].value_counts().head(10)
plt.figure()
plt.bar(top_movies.index, top_movies.values)
plt.title("Топ рекомендованных фильмов")
plt.xlabel("Фильмы")
plt.ylabel("Количество рекомендаций")
plt.xticks(rotation=60)
plt.show()