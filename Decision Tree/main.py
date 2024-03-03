import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        # Node klassi: qo'shiladigan daraja, orqa, oldingi yoki bu yerda yozilgan qiymat
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        # Agar bu yashirin bo'lsa, True qaytaradi
        return self.value is not None

class DecisionTree:
    def __init__(self, min_samples=2, max_depth=100, n_features=None):
        # Kararli daraxt modeli uchun parametrlar
        self.min_samples = min_samples
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    def fit(self, X, y):
        # Modelni o'qitish
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        # Daraxtni o'stirish (rekursiv)
        n_samples, n_features = X.shape
        n_labels = np.unique(y)

        # Qo'shiladigan yashirin, darajani chegaralash yoki minimal ma'lumotlar soni qancha
        if depth >= self.max_depth or len(n_labels) == 1 or n_samples < self.min_samples:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # Chegaralanadigan xususiyatlarni tanlash
        feature_indices = np.random.choice(n_features, self.n_features, replace=False)

        best_feature, best_threshold = self._best_split(X, y, feature_indices)

        # Optimal bo'lgan daraja bo'yicha datalarni bo'lish
        left_idxs, right_idxs = self._split(X[:, best_feature], best_threshold)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(best_feature, best_threshold, left, right)

    def _most_common_label(self, y):
        # Eng ko'p uchraydigan qiymatni topish
        counter = Counter(y)
        most_common_label = counter.most_common(1)[0][0]
        return most_common_label

    def _best_split(self, X, y, feature_indices):
        # Eng yaxshi bo'lgan xususiyatni va darajani topish
        best_gain = -1
        best_feature, best_threshold = None, None

        for feature_index in feature_indices:
            thresholds = np.unique(X[:, feature_index])

            for threshold in thresholds:
                gain = self._information_gain(y, X[:, feature_index], threshold)

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_index
                    best_threshold = threshold

        return best_feature, best_threshold

    def _information_gain(self, y, X_column, threshold):
        # Ma'lumotlarni bo'lmish chiziqlar yordamida bo'lishning foydasini hisoblash
        parent_entropy = self._entropy(y)

        left_idxs, right_idxs = self._split(X_column, threshold)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        information_gain = parent_entropy - child_entropy
        return information_gain

    def _entropy(self, y):
        # Ma'lumotlar entropiyasini hisoblash
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

    def _split(self, X_column, threshold):
        # Qiymatlarni bo'lish
        left_idxs = np.argwhere(X_column <= threshold).flatten()
        right_idxs = np.argwhere(X_column > threshold).flatten()
        return left_idxs, right_idxs

    def predict(self, X):
        # Model orqali aniqlash
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        # Daraxtni doimiy ravishda o'zlashtirish
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

if __name__ == "__main__":
    # Ma'lumotlar olish
    data = load_breast_cancer()
    X, y = data.data, data.target

    # O'qitish va test qilish uchun ma'lumotlarni bo'lib bo'lish
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )

    # Modelni yaratish va o'qitish
    model = DecisionTree()
    model.fit(X_train, y_train)

    # Bashoratlarni aniqlash
    predictions = model.predict(X_test)

    # Aniqlilikni baholash
    def accuracy(y_test, y_pred):
        return np.sum(y_test == y_pred) / len(y_test) * 100

    acc = accuracy(y_test, predictions)
    print("Accuracy:", acc)
