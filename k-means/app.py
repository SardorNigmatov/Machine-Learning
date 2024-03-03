import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

# Ma'lumotlarni olish va ko'rsatish
dataset = pd.read_csv("./Mall_Customers.csv")
print(dataset.head(10))

# Qanday ma'lumotlarni ishlatishni tanlash
x = dataset.iloc[:, [3, 4]].values
print("X:", x)

# Elbow Method uchun WCSS (Within-Cluster Sum of Squares) ni hisoblash
wcss_list = []
for i in range(1, 11):
    model = KMeans(n_clusters=i, init='k-means++', random_state=42)
    model.fit(x)
    wcss_list.append(model.inertia_)

# Elbow Method grafik
plt.plot(range(1, 11), wcss_list)
plt.title("Elbow Method Grafiki")
plt.xlabel("Clusterlar soni")
plt.ylabel("WCSS (Clusterlar ichidagi kvadratlar yig'indisi)")
plt.show()

# KMeans modelni ishlatib, clusteringni amalga oshirish
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_predict = kmeans.fit_predict(x)

# Natijalarni ko'rsatish
plt.scatter(x[y_predict == 0, 0], x[y_predict == 0, 1], s=100, c='blue', label='Cluster 1')
plt.scatter(x[y_predict == 1, 0], x[y_predict == 1, 1], s=100, c='green', label='Cluster 2')
plt.scatter(x[y_predict == 2, 0], x[y_predict == 2, 1], s=100, c='red', label='Cluster 3')
plt.scatter(x[y_predict == 3, 0], x[y_predict == 3, 1], s=100, c='cyan', label='Cluster 4')
plt.scatter(x[y_predict == 4, 0], x[y_predict == 4, 1], s=100, c='magenta', label='Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', label='Centroid')
plt.title('Mijozlar guruhlari')
plt.xlabel('Yillik daromad (k$)')
plt.ylabel('Xarajat ko\'rsatkichi (1-100)')
plt.legend()
plt.show()
