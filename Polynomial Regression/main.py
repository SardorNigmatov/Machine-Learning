import numpy as np
import matplotlib.pyplot as plt

class PolynomialRegression:
    def __init__(self, degree):
        self.degree = degree
        self.theta = None

    def _create_polynomial_matrix(self, X):
        # Polinomial matritsani yaratish
        X_poly = np.ones((len(X), 1)) 
        for d in range(1, self.degree + 1):
            X_poly = np.c_[X_poly, X**d]
        return X_poly

    def fit(self, X, y):
        # Modelni o'qitish
        X_poly = self._create_polynomial_matrix(X)
        self.theta = np.linalg.inv(X_poly.T @ X_poly) @ X_poly.T @ y

    def predict(self, X):
        # Bashoratlar hisoblash
        X_poly = self._create_polynomial_matrix(X)
        return X_poly @ self.theta


np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 3 * X**2 + 5 * X + 2 + np.random.randn(100, 1)

# PolynomialRegression klasini darajasi 2 bo'lgan obyekt bilan yaratish
poly_reg = PolynomialRegression(degree=2)

# Modelni o'qitish
poly_reg.fit(X, y)

# Bashoratlar generatsiya qilish
X_new = np.linspace(0, 2, 100).reshape(-1, 1)
y_pred = poly_reg.predict(X_new)

# Natijalarni chizish
plt.scatter(X, y, label='Original Data')
plt.plot(X_new, y_pred, 'r-', label='Polynomial Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
