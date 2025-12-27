import numpy as np

class GDA:
    def __init__(self, reg_param = 1e-6):
        self.reg_param = reg_param
        self.classes_ = None
        self.means_ = None
        self.priors_ = None
        self.cov_ = None
        self.inv_cov_ = None
        self.log_det_cov_ = None

    def fit(self, X, y, class_labels = None):
        self.classes_ = np.unique(y)
        self.class_labels = class_labels

        n_classes = len(self.classes_)
        n_sample, n_features = X.shape
        
        self.means_ = np.zeros((n_classes, n_features))
        self.priors_ = np.zeros(n_classes)
        self.cov_ = np.zeros((n_features, n_features))

        for i,c in enumerate(self.classes_):
            X_c = X[y == c]

            self.means_[i, :] = np.mean(X_c, axis=0)
            self.priors_[i] = X_c.shape[0] / n_sample
            diff =  X_c - self.means_[i]
            self.cov_ += np.dot(diff.T, diff) 

        self.cov_ /= n_sample
        self.cov_ += self.reg_param * np.eye(n_features)

        self.inv_cov_ = np.linalg.inv(self.cov_)
        self.log_det_cov_ = np.log(np.linalg.det(self.cov_))

        return self

    def _log_gaussian(self, x, mean):
        diff = x - mean
        return (
            -0.5 * diff.T @ self.inv_cov_ @ diff
            -0.5 * self.log_det_cov_
            -0.5 * len(x) * np.log(2 * np.pi)
        )

    def _predict_one(self, x):
        posteriors = []

        for idx in range(len(self.classes_)):
            log_likelihood = self._log_gaussian(x, self.means_[idx])
            log_prior = np.log(self.priors_[idx])
            posteriors.append(log_likelihood + log_prior)

        idx_max = self.classes_[np.argmax(posteriors)]

        if self.class_labels is not None:
            return self.class_labels[idx_max]
        return self.classes_[idx_max]
    
    def predict(self, X):
        X = np.array(X)

        if X.ndim == 1:
            return self._predict_one(X)
        
        predictions = [self._predict_one(x) for x in X]
        
        return np.array(predictions)
    
    def predict_proba(self, X):
        X = np.asarray(X)
        probs = []

        for x in X:
            log_posteriors = np.array([
                self._log_gaussian(x, self.means_[idx]) + np.log(self.priors_[idx])
                for idx in range(len(self.classes_))
            ])

            log_posteriors -= np.max(log_posteriors)
            probs.append(np.exp(log_posteriors) / np.sum(np.exp(log_posteriors)))

        return np.array(probs)