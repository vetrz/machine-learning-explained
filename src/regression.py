import numpy as np

class LinearRegression_:
    def __init__(self, random_state=42):
        self._coef = None
        self._intercept = None
        self.random_state = random_state
        self._rng = np.random.default_rng(self.random_state)
        self.cost_history = []

    def _hypothesis(self, X, theta):
        return np.dot(X, theta)
  
    def _compute_step(self, X, y, theta, alpha):
        m = y.shape[0]
        predictions = self._hypothesis(X, theta)
        errors = predictions - y

        gradient = (1/m) * X.T.dot(errors)
        cost = (1 / (2 * m)) * np.sum(errors**2)

        theta = theta - alpha * gradient

        return theta, cost

    def fit(self, X, y, num_iters=800, alpha=0.05):
        m, n = X.shape
        theta = self._rng.standard_normal(n+1) * np.sqrt(1/n)

        X_b = np.c_[np.ones(m), X]
        y_b = y.ravel()

        for _ in range(num_iters):
            theta, cost = self._compute_step(X_b, y_b, theta, alpha)
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