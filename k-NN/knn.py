import numpy as np
from collections import Counter
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap

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

# Iris datasetini yuklash
iris = datasets.load_iris()
X, y = iris.data, iris.target

# Datset ning 20% ni testlash uchun ajratib olish 80% traing uchun
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Datasetni vizulalizatsiya qilish
plt.figure()
plt.scatter(X[:, 2], X[:, 3], c=y, cmap=ListedColormap(['#FF0000', '#00FF00', '#0000FF']), edgecolors='k', s=30)
plt.show()

# Klassdan obyekt olish qo'shnilar soni k = 3 ta
clf = KNN(k=3)
clf.fit(X_train, y_train)

# Bashoratlash
predictions = clf.predict(X_test)

# Confusion Matrix ni hisoblash 
conf_matrix = confusion_matrix(y_test, predictions)

# Confusion Matrizni chizish
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel('Bashorat qiymat')
plt.ylabel('Haqiqiy qiymat')
plt.title('Confusion Matrix')
plt.show()

# Model aniqligini hisoblash
accuracy = accuracy_score(y_test, predictions)
print("Aniqlik:", accuracy)
