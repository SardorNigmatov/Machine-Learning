import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN

# Random ma'lumotlarni generatsiya qilish
np.random.seed(0)
centroids = [[3, 3], [-3, -2], [2, -3], [0, 0]]
X, y = make_blobs(n_samples=500, centers=centroids, cluster_std=0.6)

# Ma'lumotlarni ekranga chiqarish
plt.scatter(X[:, 0], X[:, 1], marker='.')
plt.title('Boshlang\'ich ma\'lumotlar')
plt.xlabel('X o\'qi')
plt.ylabel('Y o\'qi')
plt.show()

# DBSCAN modelini yaratish va ma'lumotlarni uni bilan o'rganish
dbscan = DBSCAN(eps=0.7, min_samples=5)
dbscan.fit(X)

# DBSCAN natijalarini olish
labels_dbscan = dbscan.labels_

# Natijalarni ekranga chiqarish
plt.scatter(X[:, 0], X[:, 1], c=labels_dbscan, cmap='viridis', marker='.')
plt.title('DBSCAN Natijalari')
plt.xlabel('X o\'qi')
plt.ylabel('Y o\'qi')
plt.show()
