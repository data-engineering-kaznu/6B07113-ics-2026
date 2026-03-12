import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("Mall_Customers.csv")

print("Первые 5 строк датасета:")
print(df.head())


X = df[["Annual Income (k$)", "Spending Score (1-100)"]]


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


print("\nПодбор лучшего числа кластеров для K-Means:")
best_k = None
best_kmeans_score = -1

for k in range(2, 11):
    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_temp = kmeans_temp.fit_predict(X_scaled)
    score_temp = silhouette_score(X_scaled, labels_temp)
    print(f"k = {k}, silhouette = {score_temp:.4f}")

    if score_temp > best_kmeans_score:
        best_kmeans_score = score_temp
        best_k = k

print(f"\nЛучшее число кластеров для K-Means: {best_k}")
print(f"Лучший silhouette score для K-Means: {best_kmeans_score:.4f}")


kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)

df["KMeans_cluster"] = kmeans_labels


print("\nПодбор параметров для DBSCAN:")
best_dbscan_score = -1
best_eps = None
best_min_samples = None
best_dbscan_labels = None

eps_values = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
min_samples_values = [3, 4, 5, 6]

for eps in eps_values:
    for min_samples in min_samples_values:
        dbscan_temp = DBSCAN(eps=eps, min_samples=min_samples)
        labels_temp = dbscan_temp.fit_predict(X_scaled)

        unique_labels = set(labels_temp)

        if -1 in unique_labels:
            n_clusters = len(unique_labels) - 1
        else:
            n_clusters = len(unique_labels)

        if n_clusters >= 2:
            try:
                score_temp = silhouette_score(X_scaled, labels_temp)
                print(f"eps = {eps}, min_samples = {min_samples}, silhouette = {score_temp:.4f}")

                if score_temp > best_dbscan_score:
                    best_dbscan_score = score_temp
                    best_eps = eps
                    best_min_samples = min_samples
                    best_dbscan_labels = labels_temp
            except:
                pass


if best_dbscan_labels is not None:
    df["DBSCAN_cluster"] = best_dbscan_labels
    print(f"\nЛучшие параметры DBSCAN: eps = {best_eps}, min_samples = {best_min_samples}")
    print(f"Лучший silhouette score для DBSCAN: {best_dbscan_score:.4f}")
else:
    df["DBSCAN_cluster"] = -1
    print("\nDBSCAN не смог найти хорошее разбиение на кластеры.")


pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df["PCA1"] = X_pca[:, 0]
df["PCA2"] = X_pca[:, 1]


plt.figure(figsize=(8, 6))
plt.scatter(
    df["Annual Income (k$)"],
    df["Spending Score (1-100)"],
    c=df["KMeans_cluster"]
)
plt.title("K-Means кластеризация клиентов")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.show()


plt.figure(figsize=(8, 6))
plt.scatter(
    df["Annual Income (k$)"],
    df["Spending Score (1-100)"],
    c=df["DBSCAN_cluster"]
)
plt.title("DBSCAN кластеризация клиентов")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.show()


plt.figure(figsize=(8, 6))
plt.scatter(
    df["PCA1"],
    df["PCA2"],
    c=df["KMeans_cluster"]
)
plt.title("K-Means кластеризация после PCA")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.show()


plt.figure(figsize=(8, 6))
plt.scatter(
    df["PCA1"],
    df["PCA2"],
    c=df["DBSCAN_cluster"]
)
plt.title("DBSCAN кластеризация после PCA")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.show()


print("\nИТОГ:")
print(f"K-Means silhouette score: {best_kmeans_score:.4f}")

if best_dbscan_labels is not None:
    print(f"DBSCAN silhouette score: {best_dbscan_score:.4f}")
else:
    print("DBSCAN silhouette score: не удалось вычислить")