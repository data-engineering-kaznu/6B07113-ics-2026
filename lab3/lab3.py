import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("fashion-mnist_test.csv")

print("Размер таблицы:", df.shape)
print("Названия первых столбцов:", df.columns[:10].tolist())

y = df["label"].values
X = df.drop("label", axis=1).values

print("Количество объектов:", X.shape[0])
print("Количество признаков:", X.shape[1])
print("Уникальные классы:", np.unique(y))


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("Размерность после PCA:", X_pca.shape)
print("Объясненная дисперсия:", pca.explained_variance_ratio_)


kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)

silhouette_kmeans = silhouette_score(X_scaled, kmeans_labels)

print("\n--- K-means ---")
print("Количество кластеров:", len(np.unique(kmeans_labels)))
print("Silhouette score:", round(silhouette_kmeans, 4))


dbscan = DBSCAN(eps=1.8, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_pca)

n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_noise = np.sum(dbscan_labels == -1)

print("\n--- DBSCAN ---")
print("Количество кластеров:", n_clusters_dbscan)
print("Количество шумовых точек:", n_noise)

mask = dbscan_labels != -1
if len(set(dbscan_labels[mask])) > 1:
    silhouette_dbscan = silhouette_score(X_pca[mask], dbscan_labels[mask])
    print("Silhouette score:", round(silhouette_dbscan, 4))
else:
    print("Silhouette score для DBSCAN посчитать нельзя")


plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, s=10)
plt.title("Fashion-MNIST: реальные классы после PCA")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.grid(True)
plt.show()


plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, s=10)
plt.title("Fashion-MNIST: кластеры K-means")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.grid(True)
plt.show()


plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=dbscan_labels, s=10)
plt.title("Fashion-MNIST: кластеры DBSCAN")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.grid(True)
plt.show()


plt.figure(figsize=(10, 4))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X[i].reshape(28, 28), cmap="gray")
    plt.title(f"Label: {y[i]}")
    plt.axis("off")
plt.suptitle("Примеры изображений Fashion-MNIST")
plt.tight_layout()
plt.show()