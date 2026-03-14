import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from math import sqrt

ratings_cols = ['user_id', 'item_id', 'rating', 'timestamp']
ratings = pd.read_csv('u.data', sep='\t', names=ratings_cols)

movies_cols = ['item_id', 'title'] + [f'genre_{i}' for i in range(19)]
movies = pd.read_csv('u.item', sep='|', encoding='latin-1', header=None)

movies = movies.iloc[:, [0, 1] + list(range(5, 24))]
movies.columns = ['item_id', 'title'] + [f'genre_{i}' for i in range(19)]
movies.columns = movies_cols

user_item_matrix = ratings.pivot_table(index='user_id', columns='item_id', values='rating')
user_similarity = cosine_similarity(user_item_matrix.fillna(0))

def predict_rating(user_id, item_id):
    sim_scores = user_similarity[user_id-1]
    item_ratings = user_item_matrix[item_id]
    mask = ~item_ratings.isna()
    if mask.sum() == 0:
        return 0
    return np.dot(sim_scores[mask], item_ratings[mask]) / sim_scores[mask].sum()

test_data = ratings.sample(1000, random_state=42)
predictions = []
real = []

for row in test_data.itertuples():
    pred = predict_rating(row.user_id, row.item_id)
    predictions.append(pred)
    real.append(row.rating)

rmse = sqrt(mean_squared_error(real, predictions))
print("RMSE (User-Based):", rmse)

genre_cols = [col for col in movies.columns if 'genre_' in col]
genre_matrix = movies.set_index('item_id')[genre_cols]

item_similarity = cosine_similarity(genre_matrix)

def recommend_content(user_id, top_n=10):
    user_ratings = user_item_matrix.loc[user_id].dropna()
    scores = np.zeros(len(genre_matrix))

    for item_id, rating in user_ratings.items():
        scores += item_similarity[item_id-1] * rating

    recommended_ids = np.argsort(scores)[::-1]
    return recommended_ids[:top_n]

user_id = 5
rec_ids = recommend_content(user_id)
rec_titles = movies[movies['item_id'].isin(rec_ids)]['title']
plt.figure(figsize=(8,4))
plt.barh(rec_titles[:10], range(10))
plt.title(f"Рекомендации для пользователя {user_id}")
plt.xlabel("Ранг рекомендации")
plt.gca().invert_yaxis()
plt.show()