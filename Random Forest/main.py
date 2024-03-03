import numpy as np
from DecisionTree import DecisionTree
from collections import Counter
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

class RandomForest:
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, n_features=None):
        # Random Forest klassi
        # n_trees: qancha desision trees ishlaydi
        # max_depth: desision tree larining maksimal o'lchami
        # min_samples_split: minimum ko'rsatkich, X ni split qilish uchun
        # n_features: biron bir tree ni fit qilish uchun kerak bo'lgan feature lar soni
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees = []

    def fit(self, X, y):
        # Random Forest ni train qilish uchun
        self.trees = []
        for _ in range(self.n_trees):
            # Har bir tree ni yaratamiz
            tree = DecisionTree(min_samples=self.min_samples_split,
                                max_depth=self.max_depth,
                                n_features=self.n_features)
            # Bootstrap samplyar orqali tree ni train qilamiz
            X_sample, y_sample = self._bootstrap_samples(X, y)
            tree.fit(X_sample, y_sample)
            # Train qilingan tree ni listga qo'shamiz
            self.trees.append(tree)

    def _bootstrap_samples(self, X, y):
        # Bootstrap samplyar orqali X va y ni split qilish
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]

    def _most_common_label(self, y):
        # List ichidagi eng ko'p takrorlangan elementni topish
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common

    def predict(self, X):
        # Random Forest ni orqali X ga bog'liq y_pred qiymatlarini hisoblash
        predictions = np.array([tree.predict(X) for tree in self.trees])
        # Predictions ni transpose qilish
        tree_preds = np.swapaxes(predictions, 0, 1)
        # Har bir sample uchun eng ko'p takrorlangan label ni hisoblash
        predictions = np.array([self._most_common_label(pred) for pred in tree_preds])
        return predictions


# Breast cancer dataset ni yuklab olish
data = load_breast_cancer()
X = data.data
y = data.target

# Train va test bo'limlariga ajratish
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)

def accuracy(y_true, y_pred):
    # Accuracy ni hisoblash
    accuracy = np.sum(y_true == y_pred) / len(y_true) * 100
    return accuracy

# Random Forest modelini yaratib, train qilib, va test qilish
model = RandomForest()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

# Accuracy ni hisoblash va chiqarish
acc = accuracy(y_test, predictions)

print("Accuracy:", acc)
