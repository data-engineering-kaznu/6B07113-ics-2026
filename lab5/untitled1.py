import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# --------------------------
# 1. Загрузка данных
# --------------------------

ratings = pd.read_csv("BX-Book-Ratings.csv", sep=";", encoding="latin-1", on_bad_lines="skip")
books = pd.read_csv("BX-Books.csv", sep=";", encoding="latin-1", on_bad_lines="skip")

ratings = ratings[['User-ID','ISBN','Book-Rating']]
books = books[['ISBN','Book-Title','Book-Author','Publisher']]

# --------------------------
# 2. Работаем с подвыборкой
# --------------------------

# берем случайные 10000 записей, чтобы не перегружать память
ratings = ratings.sample(10000, random_state=42)

# убираем нулевые оценки (0 = нет оценки)
ratings = ratings[ratings['Book-Rating'] > 0]

# объединяем с названиями книг
data = pd.merge(ratings, books, on='ISBN', how='left')

print("Размер данных:", data.shape)
print(data.head())

# --------------------------
# 3. Матрица User × Book
# --------------------------

ratings_matrix = data.pivot_table(
    index='User-ID',
    columns='Book-Title',
    values='Book-Rating'
).fillna(0)

print("Размер матрицы:", ratings_matrix.shape)

# --------------------------
# 4. Collaborative Filtering
# --------------------------

# сходство пользователей
user_similarity = cosine_similarity(ratings_matrix)
user_similarity_df = pd.DataFrame(
    user_similarity,
    index=ratings_matrix.index,
    columns=ratings_matrix.index
)

def recommend_books(user_id, n=5):
    # 5 самых похожих пользователей
    similar_users = user_similarity_df[user_id].sort_values(ascending=False)[1:6]
    
    # их оценки
    similar_users_ratings = ratings_matrix.loc[similar_users.index]
    
    # средние оценки книг
    mean_ratings = similar_users_ratings.mean()
    
    # топ-N рекомендаций
    rec = mean_ratings.sort_values(ascending=False).head(n)
    
    return rec

# пример пользователя
user_example = ratings_matrix.index[0]
collab_recs = recommend_books(user_example)
print("\nCollaborative рекомендации для пользователя:", user_example)
print(collab_recs)

# --------------------------
# 5. Метрика RMSE
# --------------------------

train, test = train_test_split(data, test_size=0.2, random_state=42)

mean_rating = train['Book-Rating'].mean()
predictions = np.full(len(test), mean_rating)

rmse = np.sqrt(mean_squared_error(test['Book-Rating'], predictions))
print("\nRMSE =", rmse)

# --------------------------
# 6. Визуализация рекомендаций
# --------------------------

plt.figure(figsize=(8,5))
collab_recs.plot(kind='bar', color='blue')
plt.title(f"Top-{len(collab_recs)} Recommended Books (User {user_example})")
plt.xlabel("Book Title")
plt.ylabel("Predicted Rating")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()