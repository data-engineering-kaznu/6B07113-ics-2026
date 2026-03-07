import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
data = pd.read_csv('input4.csv')
X = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kmeans = KMeans(n_clusters=5, random_state=42)
data['KMeans_Cluster'] = kmeans.fit_predict(X_scaled)
dbscan = DBSCAN(eps=1.5, min_samples=3)
data['DBSCAN_Cluster'] = dbscan.fit_predict(X_scaled)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
data['PCA1'] = X_pca[:, 0]
data['PCA2'] = X_pca[:, 1]
kmeans_silhouette = silhouette_score(X_scaled, data['KMeans_Cluster'])
dbscan_silhouette = silhouette_score(X_scaled, data['DBSCAN_Cluster']) if len(set(data['DBSCAN_Cluster'])) > 1 else -1
print(f'Silhouette Score KMeans: {kmeans_silhouette:.2f}')
print(f'Silhouette Score DBSCAN: {dbscan_silhouette:.2f}')
plt.figure(figsize=(8,5))
plt.scatter(data['PCA1'], data['PCA2'], c=data['KMeans_Cluster'], cmap='tab10', s=50)
plt.title('KMeans Clusters (PCA projection)')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.colorbar(label='Cluster')
plt.tight_layout()
plt.savefig('KMeans_clusters.png')  
plt.close()
plt.figure(figsize=(8,5))
plt.scatter(data['PCA1'], data['PCA2'], c=data['DBSCAN_Cluster'], cmap='tab10', s=50)
plt.title('DBSCAN Clusters (PCA projection)')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.colorbar(label='Cluster')
plt.tight_layout()
plt.savefig('DBSCAN_clusters.png')  
plt.close()
data.to_csv('output4.csv', index=False)