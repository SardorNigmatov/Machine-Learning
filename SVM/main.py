import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class SVM:
    def __init__(self, learning_rate=0.01, lambda_param=0.01, n_iters=1000):
        """
        Constructor for SVM class.

        Parameters:
        - learning_rate: The learning rate for the optimization algorithm.
        - lambda_param: Regularization parameter.
        - n_iters: Number of iterations for training.
        """
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        """
        Fit the SVM model to the training data.

        Parameters:
        - X: Training features.
        - y: Training labels.
        """
        n_samples, n_features = X.shape
        y_ = np.where(y <= 0, -1, 1)
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]

    def predict(self, X):
        """
        Predict labels for input data.

        Parameters:
        - X: Input features.

        Returns:
        - Array of predicted labels.
        """
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)

if __name__ == "__main__":
    # Generate sample data
    X, y = datasets.make_blobs(
        n_samples=50, n_features=2, centers=2, cluster_std=1.05, random_state=40
    )
    y = np.where(y == 0, -1, 1)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123
    )

    # Create SVM model instance
    model = SVM()

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    predictions = model.predict(X_test)

    def accuracy(y_true, y_pred):
        """
        Calculate accuracy of predictions.

        Parameters:
        - y_true: True labels.
        - y_pred: Predicted labels.

        Returns:
        - Accuracy percentage.
        """
        accuracy = np.sum(y_true == y_pred) / len(y_true) * 100
        return accuracy

    print("SVM classification accuracy:", accuracy(y_test, predictions))

    def visualize_svm():
        """
        Visualize SVM decision boundary.
        """
        def get_hyperplane_value(x, w, b, offset):
            return (-w[0] * x + b + offset) / w[1]

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.scatter(X[:, 0], X[:, 1], marker="o", c=y)

        x0_1 = np.amin(X[:, 0])
        x0_2 = np.amax(X[:, 0])

        x1_1 = get_hyperplane_value(x0_1, model.w, model.b, 0)
        x1_2 = get_hyperplane_value(x0_2, model.w, model.b, 0)

        x1_1_m = get_hyperplane_value(x0_1, model.w, model.b, -1)
        x1_2_m = get_hyperplane_value(x0_2, model.w, model.b, -1)

        x1_1_p = get_hyperplane_value(x0_1, model.w, model.b, 1)
        x1_2_p = get_hyperplane_value(x0_2, model.w, model.b, 1)

        ax.plot([x0_1, x0_2], [x1_1, x1_2], "y--")
        ax.plot([x0_1, x0_2], [x1_1_m, x1_2_m], "k")
        ax.plot([x0_1, x0_2], [x1_1_p, x1_2_p], "k")

        plt.show()

    visualize_svm()
