import numpy as np

class GDA:
    def __init__(self):
        self.priors = None
        self.means = None
        self.covariance = None
        self.classes = None
        self.class_labels = None

    def fit(self, X, y, class_labels = None):
        self.classes = np.unique(y)
        self.class_labels = class_labels

        n_classes = len(self.classes)
        n_sample, n_features = X.shape
        
        self.means = np.zeros((n_classes, n_features))
        self.priors = np.zeros(n_classes)
        self.covariance = np.zeros((n_features, n_features))

        for i,c in enumerate(self.classes):
            X_c = X[y == c]

            self.means[i, :] = np.mean(X_c, axis=0)
            self.priors[i] = X_c.shape[0] / n_sample
            diff =  X_c - self.means[i]
            self.covariance += np.dot(diff.T, diff) 

        self.covariance /= n_sample

        return self

    def _gaussian_density(self, x, mean, cov):
        size = len(x)
        det = np.linalg.det(cov)
        inv = np.linalg.inv(cov)

        norm_const = 1.0 / (np.power((2 * np.pi), size / 2) * np.sqrt(det))
        diff = x - mean
        exponent = -(1/2) * np.dot(np.dot(diff.T, inv), diff)

        return norm_const * np.exp(exponent)
    
    def _predict_one(self, x):
        x = np.array(x)
        
        posteriors = []
        inv_cov = np.linalg.inv(self.covariance)

        for i, _ in enumerate(self.classes):
            log_prior = np.log(self.priors[i])
            
            diff = x - self.means[i]
            log_likelihood = -0.5 * diff.T @ inv_cov @ diff
            
            posteriors.append(log_likelihood + log_prior)

        idx_max = np.argmax(posteriors)

        if self.class_labels is not None:
            return self.class_labels[idx_max]
        return self.classes[idx_max]
    
    def predict(self, X:list):
        X = np.array(X)

        if X.ndim == 1:
            return self._predict_one(X)
        
        predictions = [self._predict_one(x) for x in X]
        
        return np.array(predictions)