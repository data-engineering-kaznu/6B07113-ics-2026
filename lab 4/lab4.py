import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import pandas as pd


# 1. Загрузка данных [cite: 16]
digits = load_digits()
X, y = digits.data, digits.target

# Масштабирование признаков
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Метод «локтя» для поиска оптимального K [cite: 8]
inertia = []
k_range = range(1, 15)
for k in k_range:
    model = KMeans(n_clusters=k, n_init=10, random_state=42)
    model.fit(X_scaled)
    inertia.append(model.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(k_range, inertia, 'bo-')
plt.xlabel('Количество кластеров (k)')
plt.ylabel('Инерция')
plt.title('Метод локтя для определения оптимального k')
plt.grid(True)
plt.show()

#

# 3. Снижение размерности: PCA vs t-SNE
# PCA для сохранения глобальной структуры
pca_2d = PCA(n_components=2).fit_transform(X_scaled)
# t-SNE для лучшей визуализации локальных групп
tsne_2d = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X_scaled)

# 4. Кластеризация и оценка [cite: 17, 19]
kmeans = KMeans(n_clusters=10, n_init=10, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)

# Настройка параметров DBSCAN для масштабированных данных Digits
# eps: радиус окружности вокруг точки
# min_samples: минимум точек в радиусе, чтобы считать область "плотной"
dbscan = DBSCAN(eps=3.8, min_samples=3)
dbscan_labels = dbscan.fit_predict(X_scaled)

# Считаем количество найденных кластеров (исключая шум -1)
n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
print(f"DBSCAN нашел кластеров: {n_clusters_dbscan}")

# Расчет силуэта для оценки качества (согласно заданию [cite: 19])
if n_clusters_dbscan > 1:
    score_dbscan = silhouette_score(X_scaled, dbscan_labels)
    print(f"Silhouette Score (DBSCAN): {score_dbscan:.4f}")
else:
    print("DBSCAN: Недостаточно кластеров для расчета силуэта (увеличьте eps)")

# Расчет метрики силуэта
sil_kmeans = silhouette_score(X_scaled, kmeans_labels)
print(f"Silhouette Score (K-means): {sil_kmeans:.4f}")

# 5. Итоговая визуализация (t-SNE дает более четкую картину) [cite: 18]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# K-means на проекции t-SNE
scatter1 = ax1.scatter(tsne_2d[:, 0], tsne_2d[:, 1], c=kmeans_labels, cmap='tab10', s=15)
ax1.set_title(f"K-means Clustering (t-SNE projection)\nSilhouette: {sil_kmeans:.3f}")
plt.colorbar(scatter1, ax=ax1)

# DBSCAN на проекции t-SNE
scatter2 = ax2.scatter(tsne_2d[:, 0], tsne_2d[:, 1], c=dbscan_labels, cmap='viridis', s=15)
ax2.set_title("DBSCAN Clustering (t-SNE projection)")
plt.colorbar(scatter2, ax=ax2)

plt.tight_layout()
plt.show()

#


# Создаем DataFrame для экспорта
# Сохраняем первые несколько признаков, реальную метку и результаты обоих алгоритмов
results = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
results['true_label'] = y
results['kmeans_cluster'] = kmeans_labels
results['dbscan_cluster'] = dbscan_labels

# Сохранение в файл
results.to_csv('output.csv', index=False)
print("Результаты успешно сохранены в файл output.csv")

