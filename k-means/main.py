# import pandas as pd
# import numpy as np
# from sklearn.cluster import KMeans
# from sklearn.preprocessing import StandardScaler
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# # Ma'lumotlarni yuklash
# df = pd.read_csv("dataset.csv")

# # 'Customer Key' ustundagi amalni o'chirish
# df.drop('Customer Key', axis=1, inplace=True)

# # X xususiyatlarni ajratib olish
# X = df.values[:, 1:]
# X = np.nan_to_num(X)
# norm_data = StandardScaler().fit_transform(X)

# # k-means klastirizatsiyasini qo'llash
# k = 3
# k_means = KMeans(n_clusters=k, n_init=20)
# k_means.fit(norm_data)

# # Klastirlash natijalarini DataFrame ga qo'shish
# df['cluster'] = k_means.labels_

# # Klastirlash natijalari bo'yicha o'rtacha qiymatlar
# print(df.groupby('cluster').mean())

# # 2D nuqta chizish
# plt.figure(figsize=(8, 6))
# plt.scatter(X[:, 2], X[:, 3], s=100, c=k_means.labels_.astype(float), alpha=0.5)
# plt.xlabel('Total_visits_bank', fontsize=16)
# plt.ylabel('Total_Credit_Cards', fontsize=18)

# # Centroidlarni chizish
# centroids = k_means.cluster_centers_
# plt.scatter(centroids[:, 2], centroids[:, 3], marker='x', s=200, linewidths=3, color='red', label="Centroidlar")
# plt.legend()

# plt.title("2D Chiziqli To'plam natijalari va Centroidlar")
# plt.show()

# # 3D nuqta chizish
# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111, projection='3d')
# ax.set_xlabel('Total_Credit_Cards')
# ax.set_ylabel('Total_visits_bank')
# ax.set_zlabel('Total_visits_online')

# # Ma'lumot nuqtalarni chizish
# scatter = ax.scatter(X[:, 2], X[:, 3], X[:, 0], c=k_means.labels_.astype(float))

# # Centroidlarni chizish
# ax.scatter(centroids[:, 2], centroids[:, 3], centroids[:, 0], marker='x', s=200, linewidths=3, color='red', label="Centroidlar")
# ax.legend()

# plt.title("3D Chiziqli To'plam natijalari va Centroidlar")
# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans

# def generate_random_data(num_points=100, num_features=2):
#     np.random.seed(42)
#     return np.random.rand(num_points, num_features) * 10

# def kmeans_clustering(data, n_clusters=3):
#     kmeans = KMeans(n_clusters=n_clusters)
#     kmeans.fit(data)
#     return kmeans.cluster_centers_, kmeans.labels_

# def plot_kmeans_results(data, centroids, labels):
#     colors = ["g.", "r.", "b.", "c.", "m.", "y."]
#     for i in range(len(data)):
#         plt.plot(data[i][0], data[i][1], colors[labels[i] % len(colors)], markersize=10)

#     plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=150, linewidths=5, zorder=10)
#     plt.title('K-means natijalari')
#     plt.xlabel('X o\'qi')
#     plt.ylabel('Y o\'qi')
#     plt.show()

# def main():
#     cluster_count = 3
#     data = generate_random_data()
#     centroids, labels = kmeans_clustering(data, cluster_count)
#     plot_kmeans_results(data, centroids, labels)

# if __name__ == "__main__":
#     main()

# Kerakli kutubxonalar
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans
# from sklearn.datasets import make_blobs

# # Random ma'lumotlarni generatsiya qilish
# np.random.seed(0)
# centroids = [[3, 3], [-3, -2], [2, -3], [0, 0]]
# X, y = make_blobs(n_samples=5000, centers=centroids, cluster_std=0.8)

# # Ma'lumotlarni ekranga chiqarish
# plt.scatter(X[:, 0], X[:, 1], marker='.')
# plt.title('Boshlang\'ich ma\'lumotlar')
# plt.xlabel('X o\'qi')
# plt.ylabel('Y o\'qi')
# plt.show()

# # K-means modelini yaratish va ma'lumotlarni uni bilan o'rganish
# k_means = KMeans(init="k-means++", n_clusters=4, n_init=15)
# k_means.fit(X)

# # K-means natijalarini olish
# centroids_kmeans = k_means.cluster_centers_
# labels_kmeans = k_means.labels_

# # Natijalarni ekranga chiqarish
# fig = plt.figure(figsize=(8, 8))
# colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means.labels_))))
# ax = fig.add_subplot(1, 1, 1)

# for k, col in zip(range(len(set(k_means.labels_))), colors):
#     my_members = (k_means.labels_ == k)
#     cluster_center = k_means.cluster_centers_[k]
#     ax.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.')
#     ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=6)

# # Title of the plot
# ax.set_title('KMeans')

# # Show the plot
# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.datasets import make_blobs
# from sklearn.cluster import DBSCAN

# # Random ma'lumotlarni generatsiya qilish
# np.random.seed(0)
# centroids = [[3, 3], [-3, -2], [2, -3], [0, 0]]
# X, y = make_blobs(n_samples=500, centers=centroids, cluster_std=0.6)

# # Ma'lumotlarni ekranga chiqarish
# plt.scatter(X[:, 0], X[:, 1], marker='.')
# plt.title('Boshlang\'ich ma\'lumotlar')
# plt.xlabel('X o\'qi')
# plt.ylabel('Y o\'qi')
# plt.show()

# # DBSCAN modelini yaratish va ma'lumotlarni uni bilan o'rganish
# dbscan = DBSCAN(eps=0.7, min_samples=5)
# dbscan.fit(X)

# # DBSCAN natijalarini olish
# labels_dbscan = dbscan.labels_

# # Natijalarni ekranga chiqarish
# plt.scatter(X[:, 0], X[:, 1], c=labels_dbscan, cmap='viridis', marker='.')
# plt.title('DBSCAN Natijalari')
# plt.xlabel('X o\'qi')
# plt.ylabel('Y o\'qi')
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Ma'lumotlarni generatsiya qilish
X, y = make_blobs(n_samples=300, centers=3, cluster_std=0.60, random_state=0)

# Ierarxik klasterlash algoritmini ishga tushirish
model = AgglomerativeClustering(n_clusters=3)
y_pred = model.fit_predict(X)

# Dendrogramma tuzish
linked = linkage(X, 'ward')  # 'ward' - Uord metod
dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
plt.title('Ierarxik Klasterlash Dendrogrammasi')
plt.xlabel('Uzaklik')
plt.ylabel('Objektlar')
plt.show()

