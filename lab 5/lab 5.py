import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from math import sqrt
import warnings

# Отключаем лишние предупреждения
warnings.filterwarnings('ignore')

# ==========================================
# 1. ЗАГРУЗКА ДАННЫХ
# ==========================================
print("Загрузка данных...")
try:
    movies = pd.read_csv('ml-latest-small/movies.csv')
    ratings = pd.read_csv('ml-latest-small/ratings.csv')
except FileNotFoundError:
    print("Ошибка: Папка 'ml-latest-small' не найдена. Убедитесь, что архив распакован рядом со скриптом.")
    exit()

# ==========================================
# 2. КОНТЕНТНЫЙ ПОДХОД (Content-based)
# ==========================================
print("\n--- Подготовка контентной модели (по жанрам) ---")
# Очистка и подготовка жанров
movies['genres_str'] = movies['genres'].str.replace('|', ' ')

# Векторизация жанров через TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres_str'])

# Матрица сходства фильмов
cosine_sim_content = cosine_similarity(tfidf_matrix, tfidf_matrix)


def get_content_recommendations(title, top_n=5):
    if title not in movies['title'].values:
        return []
    idx = movies.index[movies['title'] == title][0]
    sim_scores = list(enumerate(cosine_sim_content[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n + 1]
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices].tolist()


# ==========================================
# 3. КОЛЛАБОРАТИВНАЯ ФИЛЬТРАЦИЯ (User-based)
# ==========================================
print("--- Подготовка коллаборативной модели (User-based) ---")
# Матрица Пользователи-Фильмы
user_movie_matrix = ratings.pivot(index='userId', columns='movieId', values='rating')

# Нормализация (центрирование вокруг нуля для каждого пользователя)
user_ratings_mean = user_movie_matrix.mean(axis=1)
matrix_norm = user_movie_matrix.sub(user_ratings_mean, axis=0).fillna(0)

# Сходство между пользователями
user_similarity = cosine_similarity(matrix_norm)
user_sim_df = pd.DataFrame(user_similarity, index=user_movie_matrix.index, columns=user_movie_matrix.index)


def predict_rating(user_id, movie_id):
    if movie_id not in user_movie_matrix.columns or user_id not in user_movie_matrix.index:
        return 3.0

    # Поиск похожих пользователей, которые оценили данный фильм
    similar_users = user_sim_df[user_id].drop(user_id).sort_values(ascending=False)
    movie_ratings = user_movie_matrix[movie_id].dropna()
    relevant_users = similar_users[similar_users.index.isin(movie_ratings.index)][:50]

    if relevant_users.empty:
        return user_ratings_mean[user_id]

    # Вычисление прогноза через средневзвешенное отклонение
    weighted_sum = np.dot(relevant_users, movie_ratings[relevant_users.index] - user_ratings_mean[relevant_users.index])
    sum_of_weights = relevant_users.abs().sum()

    if sum_of_weights == 0:
        return user_ratings_mean[user_id]

    pred = user_ratings_mean[user_id] + (weighted_sum / sum_of_weights)
    return np.clip(pred, 0.5, 5.0)


def get_collaborative_recommendations(user_id, top_n=5):
    unrated_movies = user_movie_matrix.columns[user_movie_matrix.loc[user_id].isna()]
    predictions = []
    # Ограничим поиск первыми 300 фильмами для скорости демонстрации
    for movie_id in unrated_movies[:300]:
        pred = predict_rating(user_id, movie_id)
        predictions.append((movie_id, pred))

    predictions.sort(key=lambda x: x[1], reverse=True)
    top_movie_ids = [x[0] for x in predictions[:top_n]]
    return movies[movies['movieId'].isin(top_movie_ids)]['title'].tolist()


# ==========================================
# 4. РАСЧЕТ МЕТРИК И СОХРАНЕНИЕ В CSV
# ==========================================
print("\n--- Расчет метрик и формирование output.csv ---")
# Тестовая выборка для оценки
test_sample = ratings.sample(500, random_state=42)
test_sample['predicted'] = test_sample.apply(lambda row: predict_rating(row['userId'], row['movieId']), axis=1)

# RMSE
rmse = sqrt(mean_squared_error(test_sample['rating'], test_sample['predicted']))
print(f"RMSE: {rmse:.4f}")

# Precision@K / Recall@K
K = 5
test_sample['is_relevant'] = test_sample['rating'] >= 4.0
test_sample['is_recommended'] = test_sample['predicted'] >= 4.0

tp = len(test_sample[(test_sample['is_relevant']) & (test_sample['is_recommended'])])
precision = tp / len(test_sample[test_sample['is_recommended']]) if len(
    test_sample[test_sample['is_recommended']]) > 0 else 0
recall = tp / len(test_sample[test_sample['is_relevant']]) if len(test_sample[test_sample['is_relevant']]) > 0 else 0

print(f"Precision@{K}: {precision:.4f}")
print(f"Recall@{K}: {recall:.4f}")

# Подтягиваем названия фильмов для CSV, чтобы было понятно
output_df = test_sample.merge(movies[['movieId', 'title']], on='movieId')
output_df = output_df[['userId', 'title', 'rating', 'predicted']]
output_df.to_csv('output.csv', index=False)
print("Результаты сохранены в файл 'output.csv'")

# ==========================================
# 5. ДЕМОНСТРАЦИЯ
# ==========================================
print("\n" + "=" * 50)
print("ВИЗУАЛИЗАЦИЯ РЕКОМЕНДАЦИЙ")
print("=" * 50)

# Пример 1: Контентный подход
movie_example = 'Toy Story (1995)'
print(f"\nПохоже на '{movie_example}' (Content-based):")
for i, m in enumerate(get_content_recommendations(movie_example), 1):
    print(f"  {i}. {m}")

# Пример 2: Коллаборативный подход
user_example = 1
print(f"\nРекомендовано для пользователя ID {user_example} (User-based):")
for i, m in enumerate(get_collaborative_recommendations(user_example), 1):
    print(f"  {i}. {m}")