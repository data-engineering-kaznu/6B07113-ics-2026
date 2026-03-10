import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("fashion-mnist_test.csv")

print("Размер датасета:", df.shape)

y = df["label"].values

X = df.drop("label", axis=1).values

print("Количество объектов:", X.shape[0])
print("Количество признаков:", X.shape[1])


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("Размерность после PCA:", X_pca.shape)


kmeans = KMeans(n_clusters=10, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)

silhouette_kmeans = silhouette_score(X_scaled, kmeans_labels)

print("\nK-means silhouette score:", silhouette_kmeans)


dbscan = DBSCAN(eps=1.8, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_pca)

clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
noise = list(dbscan_labels).count(-1)

print("\nDBSCAN clusters:", clusters)
print("Noise points:", noise)


mask = dbscan_labels != -1

if len(set(dbscan_labels[mask])) > 1:
    sil_dbscan = silhouette_score(X_pca[mask], dbscan_labels[mask])
    print("DBSCAN silhouette score:", sil_dbscan)


plt.figure(figsize=(7,5))
plt.scatter(X_pca[:,0], X_pca[:,1], c=y, s=10)
plt.title("Real classes (PCA)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.show()


plt.figure(figsize=(7,5))
plt.scatter(X_pca[:,0], X_pca[:,1], c=kmeans_labels, s=10)
plt.title("K-means clusters")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.show()

plt.figure(figsize=(7,5))
plt.scatter(X_pca[:,0], X_pca[:,1], c=dbscan_labels, s=10)
plt.title("DBSCAN clusters")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.show()

plt.figure(figsize=(8,4))

for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(X[i].reshape(28,28), cmap="gray")
    plt.title("Label: " + str(y[i]))
    plt.axis("off")

plt.suptitle("Examples from Fashion-MNIST")
plt.show()