import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
columns = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
    "class"
]
df = pd.read_csv("iris.data", header=None, names=columns)
X = df.drop("class", axis=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)
kmeans_silhouette = silhouette_score(X_scaled, kmeans_labels)
print("Silhouette score (K-means):", kmeans_silhouette)

dbscan = DBSCAN(eps=0.5, min_samples=4)
dbscan_labels = dbscan.fit_predict(X_scaled)
if len(set(dbscan_labels)) > 1 and -1 not in set(dbscan_labels):
    dbscan_silhouette = silhouette_score(X_scaled, dbscan_labels)
    print("Silhouette score (DBSCAN):", dbscan_silhouette)
else:
    print("Silhouette score (DBSCAN) нельзя вычислить")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure()
plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=kmeans_labels
)
plt.title("K-means кластеризация (PCA)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)
plt.show()

plt.figure()
plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=dbscan_labels
)
plt.title("DBSCAN кластеризация (PCA)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)
plt.show()
print("Метки кластеров DBSCAN:")
print(dbscan_labels)