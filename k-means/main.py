import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

def euclidean_distance(x1, x2):
    # Ikki nuqta o'rtasidagi Euclidean masofa
    return np.sqrt(np.sum((x1 - x2) ** 2))

class KMeans:   
    def __init__(self, K=5, max_iters=1000, plot_steps=False):
        # K: clusterlar soni
        # max_iters: k-means uchun maksimal takrorlashlar soni
        # plot_steps: har bir takrorlashda chizishni yoqish/chizishni o'chirish bayonoti
        self.K = K
        self.max_iters = max_iters
        self.plot_steps = plot_steps
        
        # Clusterlar va centroidlar saqlash uchun ro'yxatlar
        self.clusters = [[] for _ in range(self.K)]
        self.centroids = []

    def predict(self, X):
        # X: kiritilgan ma'lumotlar
        self.X = X
        self.n_samples, self.n_features = X.shape
        
        # Centroidlarni tasodifiy tanlash
        random_sample_idxs = np.random.choice(self.n_samples, self.K, replace=False)
        self.centroids = [self.X[idx] for idx in random_sample_idxs]

        # K-means takrorlashlar
        for _ in range(self.max_iters):
            # Samplelarni clusterlarga bo'lish
            self.clusters = self._create_clusters(self.centroids)
            
            # Har bir takrorlashda chizishni ko'rsatish (agar yoqilgan bo'lsa)
            if self.plot_steps:
                self.plot()
            
            # Eski centroidlarni saqlash va yangi centroidlarni hisoblash
            centroids_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)
            
            # Konvergentsiyani tekshirish
            if self._is_converged(centroids_old, self.centroids):
                break
        
        # Har bir sample uchun clusterlar bo'yicha labelni olish
        return self._get_cluster_labels(self.clusters)

    def _get_cluster_labels(self, clusters):
        # Har bir sample uchun cluster labelini belgilash
        labels = np.empty(self.n_samples)
        for cluster_idx, cluster in enumerate(clusters):
            for sample_index in cluster:
                labels[sample_index] = cluster_idx
        return labels

    def _create_clusters(self, centroids):
        # Har bir sample ni eng yaqin centroidga bog'lash
        clusters = [[] for _ in range(self.K)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters

    def _closest_centroid(self, sample, centroids):
        # Berilgan sample uchun eng yaqin centroidning indeksini topish
        distances = [euclidean_distance(sample, point) for point in centroids]
        closest_idx = np.argmin(distances)
        return closest_idx

    def _get_centroids(self, clusters):
        # Cluster o'rtasiga asosan yangi centroidlarni hisoblash
        centroids = np.zeros((self.K, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids

    def _is_converged(self, centroids_old, centroids):
        # Eski va yangi centroidlarni solishtirib konvergentsiyani tekshirish
        distances = [euclidean_distance(centroids_old[i], centroids[i]) for i in range(self.K)]
        return sum(distances) == 0

    def plot(self):
        # Clusterlarni va centroidlarni chizish
        fig, ax = plt.subplots(figsize=(8, 6))
        
        for i, cluster in enumerate(self.clusters):
            points = self.X[cluster].T
            ax.scatter(*points)
        
        for point in self.centroids:
            ax.scatter(*point, marker="x", color="black", linewidth=2)
        
        plt.show()


if __name__ == "__main__":
    # 3 ta clusterli sintetik ma'lumot generatsiya qilish
    np.random.seed(42)
    X, y = make_blobs(centers=3, n_samples=500, n_features=2, shuffle=True, random_state=40)
    clusters = len(np.unique(y))
    print("Clusterlar soni:", clusters)
    
    # KMeans modelni yaratish va uni boshqarish
    model = KMeans(K=clusters, max_iters=150, plot_steps=True)
    y_pred = model.predict(X)
    
    # Yakun clusterlarni va centroidlarni chizish
    model.plot()
