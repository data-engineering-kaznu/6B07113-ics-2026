import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error

# 1. Загрузка данных
movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

movies = movies[["movieId", "title", "genres"]]
ratings = ratings[["userId", "movieId", "rating"]]

print("Данные загружены")
print()

# 2. Уменьшаем датасет
top_users = ratings["userId"].value_counts().head(20).index
ratings = ratings[ratings["userId"].isin(top_users)]

top_movies = ratings["movieId"].value_counts().head(80).index
ratings = ratings[ratings["movieId"].isin(top_movies)]
movies = movies[movies["movieId"].isin(top_movies)]

print("Пользователей:", ratings["userId"].nunique())
print("Фильмов:", ratings["movieId"].nunique())
print("Оценок:", len(ratings))
print()

# 3. Делим на train и test
train_ratings, test_ratings = train_test_split(ratings, test_size=0.2, random_state=42)

# 4. Content-Based
movies["genres"] = movies["genres"].fillna("")
movies["genres_list"] = movies["genres"].apply(lambda x: x.split("|"))

all_genres = []
for row in movies["genres_list"]:
    for genre in row:
        if genre != "(no genres listed)" and genre not in all_genres:
            all_genres.append(genre)

for genre in all_genres:
    movies[genre] = movies["genres_list"].apply(lambda x: 1 if genre in x else 0)

def build_user_profile(user_id):
    user_data = train_ratings[train_ratings["userId"] == user_id]
    liked = user_data[user_data["rating"] >= 4.0]

    if liked.empty:
        return np.zeros(len(all_genres))

    liked_movies = liked.merge(movies[["movieId"] + all_genres], on="movieId")
    profile = liked_movies[all_genres].mean().values
    return profile

def recommend_content(user_id, top_n=10):
    profile = build_user_profile(user_id)
    watched = set(train_ratings[train_ratings["userId"] == user_id]["movieId"])
    candidates = movies[~movies["movieId"].isin(watched)].copy()

    if candidates.empty:
        return pd.DataFrame(columns=["movieId", "title", "score"])

    scores = candidates[all_genres].values.dot(profile)
    candidates["score"] = scores

    recs = candidates[["movieId", "title", "score"]]
    recs = recs.sort_values(by="score", ascending=False).head(top_n)
    return recs

# 5. Collaborative Filtering
user_item = train_ratings.pivot_table(index="userId", columns="movieId", values="rating")
user_item_filled = user_item.fillna(0)

similarity = cosine_similarity(user_item_filled)
similarity_df = pd.DataFrame(similarity, index=user_item.index, columns=user_item.index)

def predict_rating(user_id, movie_id):
    if user_id not in user_item.index:
        return np.nan
    if movie_id not in user_item.columns:
        return np.nan

    movie_ratings = user_item[movie_id].dropna()
    if movie_ratings.empty:
        return np.nan

    sims = similarity_df.loc[user_id, movie_ratings.index]

    if user_id in sims.index:
        sims = sims.drop(user_id, errors="ignore")
        movie_ratings = movie_ratings.drop(user_id, errors="ignore")

    sims = sims[sims > 0]
    movie_ratings = movie_ratings[sims.index]

    if sims.empty:
        return np.nan

    pred = np.dot(sims, movie_ratings) / sims.sum()
    return pred

def recommend_collab(user_id, top_n=10):
    if user_id not in user_item.index:
        return pd.DataFrame(columns=["movieId", "title", "pred_rating"])

    watched = set(user_item.loc[user_id].dropna().index)
    candidates = [m for m in user_item.columns if m not in watched]

    results = []

    for movie_id in candidates[:40]:
        pred = predict_rating(user_id, movie_id)
        if not np.isnan(pred):
            title = movies[movies["movieId"] == movie_id]["title"].values[0]
            results.append([movie_id, title, pred])

    recs = pd.DataFrame(results, columns=["movieId", "title", "pred_rating"])
    recs = recs.sort_values(by="pred_rating", ascending=False).head(top_n)
    return recs

# 6. RMSE
true_vals = []
pred_vals = []

for _, row in test_ratings.head(50).iterrows():
    pred = predict_rating(row["userId"], row["movieId"])
    if not np.isnan(pred):
        true_vals.append(row["rating"])
        pred_vals.append(pred)

if len(true_vals) > 0:
    rmse = sqrt(mean_squared_error(true_vals, pred_vals))
else:
    rmse = None

print("RMSE =", rmse)
print()

# 7. Precision@5 и Recall@5
def precision_recall_at5(user_id, rec_ids):
    user_test = test_ratings[test_ratings["userId"] == user_id]
    relevant = set(user_test[user_test["rating"] >= 4.0]["movieId"])

    if len(relevant) == 0:
        return None, None

    recommended = set(rec_ids[:5])
    hits = len(recommended & relevant)

    precision = hits / 5
    recall = hits / len(relevant)

    return precision, recall

users_eval = list(test_ratings["userId"].unique())[:5]

content_p = []
content_r = []
collab_p = []
collab_r = []

for user_id in users_eval:
    rec1 = recommend_content(user_id, top_n=5)
    rec2 = recommend_collab(user_id, top_n=5)

    p1, r1 = precision_recall_at5(user_id, rec1["movieId"].tolist())
    p2, r2 = precision_recall_at5(user_id, rec2["movieId"].tolist())

    if p1 is not None:
        content_p.append(p1)
        content_r.append(r1)

    if p2 is not None:
        collab_p.append(p2)
        collab_r.append(r2)

print("Content Precision@5 =", np.mean(content_p) if content_p else 0)
print("Content Recall@5 =", np.mean(content_r) if content_r else 0)
print()
print("Collaborative Precision@5 =", np.mean(collab_p) if collab_p else 0)
print("Collaborative Recall@5 =", np.mean(collab_r) if collab_r else 0)
print()

# 8. Рекомендации для одного пользователя
example_user = train_ratings["userId"].iloc[0]

content_recs = recommend_content(example_user, top_n=10)
collab_recs = recommend_collab(example_user, top_n=10)

print("Пользователь:", example_user)
print()

print("Content-Based рекомендации:")
print(content_recs)
print()

print("Collaborative рекомендации:")
print(collab_recs)
print()

# 9. Графики
plt.figure(figsize=(10, 5))
plt.barh(content_recs["title"].iloc[::-1], content_recs["score"].iloc[::-1])
plt.title("Content-Based Recommendations")
plt.xlabel("Score")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
plt.barh(collab_recs["title"].iloc[::-1], collab_recs["pred_rating"].iloc[::-1])
plt.title("Collaborative Recommendations")
plt.xlabel("Predicted Rating")
plt.tight_layout()
plt.show()