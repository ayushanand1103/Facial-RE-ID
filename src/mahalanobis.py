"""Regularized pooled within-identity covariance and Mahalanobis distance."""

import numpy as np
from scipy.linalg import solve_triangular


class MahalanobisMetric:
    def __init__(self, regularization=0.1):
        if not 0 < regularization <= 1:
            raise ValueError("regularization must lie in (0, 1]")
        self.regularization = regularization
        self.covariance = None

    def fit(self, features, labels):
        features = np.asarray(features, dtype=np.float64)
        labels = np.asarray(labels)
        if features.ndim != 2 or labels.shape != (len(features),):
            raise ValueError("Features and labels have incompatible shapes")
        if not np.isfinite(features).all():
            raise ValueError("Features must be finite")
        identities = np.unique(labels)
        residuals = np.empty_like(features)
        for identity in identities:
            mask = labels == identity
            if mask.sum() < 2:
                raise ValueError("Each identity needs at least two enrollment samples")
            residuals[mask] = features[mask] - features[mask].mean(axis=0)
        covariance = residuals.T @ residuals / (len(features) - len(identities))
        dimension = features.shape[1]
        scale = max(float(np.trace(covariance) / dimension), 1e-6)
        self.covariance = (
            (1 - self.regularization) * covariance
            + self.regularization * scale * np.eye(dimension)
        )
        self.cholesky = np.linalg.cholesky(self.covariance)
        return self

    def distances(self, query, gallery):
        if self.covariance is None:
            raise RuntimeError("Fit the Mahalanobis metric first")
        differences = np.asarray(gallery, dtype=float) - np.asarray(query, dtype=float)
        whitened = solve_triangular(self.cholesky, differences.T, lower=True)
        return np.linalg.norm(whitened, axis=0)
