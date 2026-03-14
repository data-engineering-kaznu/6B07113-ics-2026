import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

print("Movies:")
print(movies.head())

print("\nRatings:")
print(ratings.head())

movies = movies[['movieId', 'title', 'genres']]
ratings = ratings[['userId', 'movieId', 'rating']]

movies['genres'] = movies['genres'].fillna('')
movies['genres'] = movies['genres'].str.replace('|', ' ', regex=False)

cv = CountVectorizer()
genre_matrix = cv.fit_transform(movies['genres'])

similarity = cosine_similarity(genre_matrix)

movie_index = pd.Series(movies.index, index=movies['title']).drop_duplicates()

def recommend_by_content(title, n=5):
    if title not in movie_index:
        print("Фильм не найден")
        return

    idx = movie_index[title]
    sim_scores = list(enumerate(similarity[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:n+1]

    movie_indices = [i[0] for i in sim_scores]
    print(f"\nРекомендации для фильма '{title}':")
    print(movies.iloc[movie_indices][['title', 'genres']])

train, test = train_test_split(ratings, test_size=0.2, random_state=42)

user_movie = train.pivot_table(index='userId', columns='movieId', values='rating')
user_movie = user_movie.fillna(0)

user_similarity = cosine_similarity(user_movie)
user_similarity_df = pd.DataFrame(user_similarity, index=user_movie.index, columns=user_movie.index)

def predict_rating(user_id, movie_id):
    if user_id not in user_movie.index:
        return ratings['rating'].mean()

    if movie_id not in user_movie.columns:
        return ratings['rating'].mean()

    similar_users = user_similarity_df[user_id].sort_values(ascending=False)

    weighted_sum = 0
    sim_sum = 0

    for other_user, sim in similar_users.items():
        if other_user == user_id:
            continue

        rating = user_movie.loc[other_user, movie_id]
        if rating > 0:
            weighted_sum += sim * rating
            sim_sum += sim

    if sim_sum == 0:
        return ratings['rating'].mean()

    return weighted_sum / sim_sum

def recommend_for_user(user_id, n=5):
    if user_id not in user_movie.index:
        print("Пользователь не найден")
        return

    watched = ratings[ratings['userId'] == user_id]['movieId'].tolist()
    all_movies = movies['movieId'].tolist()

    unwatched = [movie for movie in all_movies if movie not in watched]

    predictions = []
    for movie_id in unwatched[:500]:  # ограничение для скорости
        pred = predict_rating(user_id, movie_id)
        predictions.append((movie_id, pred))

    predictions = sorted(predictions, key=lambda x: x[1], reverse=True)[:n]

    print(f"\nРекомендации для пользователя {user_id}:")
    for movie_id, score in predictions:
        title = movies[movies['movieId'] == movie_id]['title'].values[0]
        print(f"{title} -> предсказанный рейтинг: {score:.2f}")


true_ratings = []
pred_ratings = []

small_test = test.head(200)  

for _, row in small_test.iterrows():
    user_id = row['userId']
    movie_id = row['movieId']
    true_rating = row['rating']
    pred_rating = predict_rating(user_id, movie_id)

    true_ratings.append(true_rating)
    pred_ratings.append(pred_rating)

rmse = np.sqrt(mean_squared_error(true_ratings, pred_ratings))
print(f"\nRMSE: {rmse:.4f}")

recommend_by_content("Toy Story (1995)", 5)

recommend_for_user(1, 5)