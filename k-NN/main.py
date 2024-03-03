import numpy as np
from collections import Counter
from sklearn import datasets
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
import pandas as pd

def euclidean_distance(x1, x2):
    """Yevklid masofasini hisoblash

    Args:
        x1 (_type_): 1-chi nuqta kordinatasi
        x2 (_type_): 2-chi nuqta kordinatasi

    Returns:
        _type_: Nuqtalar orasidagi eng yaqin masofani qaytaradi
    """
    distance = np.sqrt(np.sum((x1 - x2) ** 2))
    return distance

class KNN:
    def __init__(self, k=3):
        """K-NN algoritmi

        Args:
            k (int, optional): Eng yaqin qo'shnilar soni. Standart holatda k = 3
        """
        self.k = k

    def fit(self, X, y):
        """Modelni o'qitish metodi

        Args:
            X (_type_): o'qtish qiymati
            y (_type_): o'qitish qiymati natijasi
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        """Modelni bashoratlash metodi

        Args:
            X (_type_): X qiymat

        Returns:
            _type_: Bashorat natijaini qaytaradi
        """
        predictions = [self._predict(x) for x in X]
        return predictions

    def _predict(self, x):
        """Nuqtaning qaysi sinfga tegishli ekanligin qaytaruvchi metod

        Args:
            x (_type_): Kiruvchi qiymat

        Returns:
            _type_: Tegishli sinfni qaytaradi
        """
        # Masofani hisoblash
        distance = [euclidean_distance(x, x_train) for x_train in self.X_train]

        # Eng yaqin qo'shnilar topish
        k_indices = np.argsort(distance)[:self.k]

        # Eng yaqin qo'shnilar label
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # Eng yaqin qo'shnilar labellarini qaytaradi
        most_common = Counter(k_nearest_labels).most_common(1)[0][0]

        return most_common
    
    def plot_confusion_matrix(self, conf_matrix):
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Bashorat qilingan")
        plt.ylabel("Haqiqiy")
        plt.title("Confusion Matrix")
        plt.show()


# Ma'lumotlar to'plamini olish
dataset = pd.read_csv("./diabetes.csv")

# Xususiyatlarni (X) va maqsad o'zgaruvchisini (y) ajratib olish
features = dataset.iloc[:, [1, 2, 3, 4, 5, 6]].values
labels = dataset.iloc[:, 8].values

# Xususiyatlarni StandardScaler yordamida normalizatsiya qilish
scaler = StandardScaler()
features_normalized = scaler.fit_transform(features)

# Normalizatsiya qilingan ma'lumotlarni o'qish va test qilish uchun ajratib olish
features_train, features_test, labels_train, labels_test = train_test_split(
    features_normalized, labels, test_size=35, random_state=1432
)

# KNN klassining yangi namunasi yaratish
model = KNN(k=7)

# Modelni o'qitish
model.fit(features_train, labels_train)

# Test ma'lumotlariga bashorat qilish
predictions = model.predict(features_test)

# Confusion matrixni hisoblash
conf_matrix = confusion_matrix(labels_test, predictions)

# Confusion matrixni ko'rsatish
print("Confusion Matrix:")
print(conf_matrix)

# Aniqlilikni hisoblash va ko'rsatish
accuracy = accuracy_score(labels_test, predictions) * 100
print(f"Accuracy: {accuracy:.2f} %")

# Confusion matrixni vizualizatsiya qilish
model.plot_confusion_matrix(conf_matrix)
