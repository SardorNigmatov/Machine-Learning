import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class QuadraticRegression:
    def __init__(self, learning_rate=0.01, epsilon=0.01, max_iterations=10000):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.weights = {'w0': 0, 'w1': 0, 'w2': 0}
        self.losses = []

    def predict(self, x):
        return self.weights['w0'] + self.weights['w1'] * x + self.weights['w2'] * (x ** 2)

    def loss(self, x, y):
        total_error = np.sum((y - self.predict(x)) ** 2)
        return total_error / len(x)

    def update_weights(self, x, y):
        N = len(x)
        y_pred = self.predict(x)

        # Gradients
        dw0 = -2 * np.sum(y - y_pred) / N
        dw1 = -2 * np.sum((y - y_pred) * x) / N
        dw2 = -2 * np.sum((y - y_pred) * (x ** 2)) / N

        # Update weights
        self.weights['w0'] -= self.learning_rate * dw0
        self.weights['w1'] -= self.learning_rate * dw1
        self.weights['w2'] -= self.learning_rate * dw2

    def train(self, x, y):
        x = np.array(x)
        y = np.array(y)
        epoch = 0

        while epoch < self.max_iterations:
            self.update_weights(x, y)
            current_loss = self.loss(x, y)
            self.losses.append(current_loss)

            if current_loss < self.epsilon:
                break

            epoch += 1

# Datasetni yaratish
x_data = [1, 2, 3, 4, 5, 6] # x ning qiymatlari
y_data = [0, 0, 2, 6, 12, 20] # y ning qiymatlari

lr = float(input("Learning rate:"))
epsilon = float(input("Epsilon:"))
max_iterations = int(input("N="))

# Modelni o'qitish va chizish
quadratic_regression = QuadraticRegression(learning_rate=lr, epsilon=epsilon, max_iterations=max_iterations)
quadratic_regression.train(x_data, y_data)

# plt.scatter(x_data,y_data)
# plt.show()

print("w0= ",quadratic_regression.weights["w0"])
print("w1= ",quadratic_regression.weights["w1"])
print("w2= ",quadratic_regression.weights["w2"])
print("Bashorat:(x=7)",quadratic_regression.predict(x=7))

# Modelni chizib olyapmiz
plt.scatter(x_data, y_data, label="Ma'lumot nuqtalari")
x_range = np.linspace(min(x_data), max(x_data), 100)
y_range = quadratic_regression.predict(x_range)
plt.plot(x_range, y_range, color="red", label="Kvadratik Regression")
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('Ma\'lumot Nuqtalari va Kvadratik Regression')
plt.show()

# Lossning o'zgarishini ko'rsatish
plt.plot(range(len(quadratic_regression.losses)), quadratic_regression.losses, label="Loss")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Lossning O\'zgarishi')
plt.show()

