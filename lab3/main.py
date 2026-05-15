import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


# 1. Загружаем данные
iris = load_iris()
X = iris.data

print("=== Лабораторная работа 3 ===")
print("Распознавание образов с помощью обучения без учителя")
print()

print("Размер данных:", X.shape)
print("Первые 5 объектов:")
print(X[:5])
print()


# 2. Масштабируем данные
# Это полезно для DBSCAN и вообще для кластеризации
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Данные после масштабирования:")
print(X_scaled[:5])
print()


# 3. K-means кластеризация
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)

print("=== K-means ===")
print("Метки кластеров:")
print(kmeans_labels)
print()

unique_kmeans, counts_kmeans = np.unique(kmeans_labels, return_counts=True)
print("Количество объектов в каждом кластере:")
for cluster, count in zip(unique_kmeans, counts_kmeans):
    print(f"Кластер {cluster}: {count} объектов")
print()

kmeans_silhouette = silhouette_score(X_scaled, kmeans_labels)
print("Silhouette Score для K-means:", round(kmeans_silhouette, 4))
print()


# 4. DBSCAN кластеризация
dbscan = DBSCAN(eps=0.8, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_scaled)

print("=== DBSCAN ===")
print("Метки кластеров:")
print(dbscan_labels)
print()

unique_dbscan, counts_dbscan = np.unique(dbscan_labels, return_counts=True)
print("Количество объектов в каждом кластере:")
for cluster, count in zip(unique_dbscan, counts_dbscan):
    if cluster == -1:
        print(f"Шум (-1): {count} объектов")
    else:
        print(f"Кластер {cluster}: {count} объектов")
print()

# Проверка: silhouette score можно считать, только если кластеров больше одного
dbscan_clusters = set(dbscan_labels)
if -1 in dbscan_clusters:
    dbscan_clusters.remove(-1)

if len(dbscan_clusters) > 1:
    dbscan_silhouette = silhouette_score(X_scaled, dbscan_labels)
    print("Silhouette Score для DBSCAN:", round(dbscan_silhouette, 4))
else:
    print("Silhouette Score для DBSCAN нельзя корректно вычислить.")
print()


# 5. PCA для уменьшения размерности до 2
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("=== PCA ===")
print("Размер данных после PCA:", X_pca.shape)
print("Первые 5 объектов после PCA:")
print(X_pca[:5])
print()


# 6. Визуализация K-means
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels)
plt.title("K-means кластеризация после PCA")
plt.xlabel("Главная компонента 1")
plt.ylabel("Главная компонента 2")
plt.grid(True)
plt.show()


# 7. Визуализация DBSCAN
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=dbscan_labels)
plt.title("DBSCAN кластеризация после PCA")
plt.xlabel("Главная компонента 1")
plt.ylabel("Главная компонента 2")
plt.grid(True)
plt.show()