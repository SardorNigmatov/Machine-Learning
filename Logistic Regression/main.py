import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets

def sigmoid(x):
    # Sigmoid funksiyasi
    clipped_x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-clipped_x))

class LogisticRegression():
    def __init__(self, lr=0.01, n_iters=1000):
        # Konstruktor funksiya
        self.lr = lr
        self.n_iters = n_iters
        self.bias = None
        self.weights = None

    def fit(self, X, y):
        # Modelni o'qitish metodi
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            # Lineyka regressiya formulasi
            linear_pred = np.dot(X, self.weights) + self.bias
            predictions = sigmoid(linear_pred)

            # Gradientni hisoblash
            dw = (1/n_samples) * np.dot(X.T, (predictions - y))
            db = (1/n_samples) * np.sum(predictions - y)

            # Model parametrlarini yangilash
            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db

    def predict(self, X):
        # Modelni bashoratlash metodi
        linear_pred = np.dot(X, self.weights) + self.bias
        y_pred = sigmoid(linear_pred)
        # Aralashtirish pragmani asosida klassni aniqlash
        class_pred = [0 if y <= 0.5 else 1 for y in y_pred]
        return class_pred

def accuracy(y_pred, y_test):
    # Aniqlilikni hisoblash
    return np.sum(y_pred == y_test) / len(y_test) * 100

# Ma'lumotlar to'plamini olish
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# Logistik regressiya modelini yaratib o'qitish
model = LogisticRegression()
model.fit(X_train, y_train)

# Test to'plamida bashorat qilish
y_pred = model.predict(X_test)

# Aniqlilikni hisoblash va chiqarish
acc = accuracy(y_pred, y_test)
print("Accuracy: {0:.2f}%".format(float(acc)))
