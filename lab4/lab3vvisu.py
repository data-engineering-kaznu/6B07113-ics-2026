import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

data = pd.read_csv("Mall_Customers.csv")

X = data[['Age','Annual Income (k$)','Spending Score (1-100)']]

# масштабирование данных
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. K-means кластеризация

kmeans = KMeans(n_clusters=5, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)

kmeans_silhouette = silhouette_score(X_scaled, kmeans_labels)

print("Silhouette score (KMeans):", kmeans_silhouette)

# 3. DBSCAN кластеризация

dbscan = DBSCAN(eps=0.7, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_scaled)

# silhouette считается только если больше 1 кластера
if len(set(dbscan_labels)) > 1:
    dbscan_silhouette = silhouette_score(X_scaled, dbscan_labels)
    print("Silhouette score (DBSCAN):", dbscan_silhouette)
else:
    print("DBSCAN не нашел достаточного количества кластеров")

# 4. PCA (уменьшение размерности)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 5. Визуализация KMeans

plt.figure()

plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=kmeans_labels
)

plt.title("KMeans Clusters (PCA)")
plt.xlabel("PC1")
plt.ylabel("PC2")

plt.show()

# 6. Визуализация DBSCAN

plt.figure()

plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=dbscan_labels
)

plt.title("DBSCAN Clusters (PCA)")
plt.xlabel("PC1")
plt.ylabel("PC2")

plt.show()