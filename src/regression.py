import numpy as np


class LinearRegression_:
    def __init__(self, random_state=42):
        self._coef = None
        self._intercept = None

        self.random_state = random_state
        self._rng = np.random.default_rng(self.random_state)

    def _hypothesis(self, X, theta):
        return np.dot(X, theta)

    def _cost(self, X, y, theta):
        m = y.shape[0]
        predictions = self._hypothesis(X, theta)
        sq_errors = (predictions - y) ** 2
        J = 1 / 2 * (1 / m) * np.sum(sq_errors)

        return J

    def _gradient_descent(self, X, y, theta, alpha):
        m = y.shape[0]
        predictions = self._hypothesis(X, theta)
        errors = predictions - y

        theta = theta - (alpha / m) * (X.T.dot(errors))

        J = 1 / 2 * (1 / m) * np.sum(errors**2)

        return theta, J

    def fit(self, X, y, num_iters=600, alpha=0.001):
        self.cost_history = []
        m, n = X.shape
        theta = self._rng.random(n + 1)

        X = np.c_[np.ones(m), X]
        y = y.ravel()

        for _ in range(num_iters):
            theta, cost = self._gradient_descent(X, y, theta, alpha)
            self.cost_history.append(cost)

        self._intercept = theta[0]
        self._coef = theta[1:]

        return self

    def predict(self, X):
        return X.dot(self._coef) + self._intercept


class LogisticRegression_:
    def __init__(self, random_state=42):
        self._coef = None
        self._intercept = None

        self.random_state = random_state
        self._rng = np.random.default_rng(self.random_state)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _log_loss(self, y, p):
        cost = -np.mean((y * np.log(p)) + ((1 - y) * np.log(1 - p)))

        return cost

    def _gradient(self, X, y, probabilities):
        m = y.shape[0]

        return (X.T @ (probabilities - y)) / m

    def fit(self, X, y, epochs=100, learning_rate=0.001):
        self._intercept = np.array([0])
        self._coef = self._rng.random(X.shape[1])

        theta = np.concatenate((self._intercept, self._coef))

        X = np.c_[np.ones(X.shape[0]), X]
        z = X @ theta
        probabilities = self._sigmoid(z)

        for _ in range(epochs):
            gradients = self._gradient(X, y, probabilities)

            theta = theta - learning_rate * gradients

            z = X @ theta
            probabilities = self._sigmoid(z)

        self._intercept, self._coef = np.array([theta[0]]), theta[1:]

        return self

    def predict_proba(self, X):
        if self._coef is None or self._intercept is None:
            raise ValueError("The model has not been trained yet, call fit() first")

        z = (X @ self._coef) + self._intercept

        return self._sigmoid(z)

    def predict(self, X, threshold=0.5):
        probabilites = self.predict_proba(X)

        return (probabilites >= threshold).astype(int)
