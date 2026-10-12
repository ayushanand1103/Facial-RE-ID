"""Eigenfaces via thin SVD, without allocating a 4096 x 4096 covariance."""

import numpy as np


class Eigenfaces:
    def __init__(self, n_components=50):
        if not isinstance(n_components, int) or n_components < 1:
            raise ValueError("n_components must be a positive integer")
        self.n_components = n_components
        self.mean = None

    def fit(self, vectors):
        samples = np.asarray(vectors, dtype=np.float64)
        if samples.ndim != 2 or samples.shape[0] < 2 or samples.shape[1] < 1 or not np.isfinite(samples).all():
            raise ValueError("PCA requires at least two finite sample vectors")
        self.mean = samples.mean(axis=0)
        centered = samples - self.mean
        _, singular_values, right_vectors = np.linalg.svd(centered, full_matrices=False)
        tolerance = max(centered.shape) * np.finfo(float).eps * singular_values[0]
        rank = int(np.sum(singular_values > tolerance))
        if rank == 0:
            raise ValueError("Enrollment images have no variation; capture different frames")
        count = min(self.n_components, rank)
        # Columns are covariance eigenvectors: U in z = U.T @ (x - mean).
        self.components = right_vectors[:count].T
        all_eigenvalues = singular_values ** 2 / (samples.shape[0] - 1)
        self.eigenvalues = all_eigenvalues[:count]
        self.explained_variance_ratio = self.eigenvalues / all_eigenvalues.sum()
        return self

    def transform(self, vectors):
        if self.mean is None:
            raise RuntimeError("Fit PCA before projecting faces")
        samples = np.asarray(vectors, dtype=np.float64)
        if samples.ndim not in (1, 2) or samples.shape[-1] != self.mean.size:
            raise ValueError("Vector dimensions do not match the fitted PCA")
        if not np.isfinite(samples).all():
            raise ValueError("PCA input must be finite")
        return (samples - self.mean) @ self.components

    def inverse_transform(self, projected):
        if self.mean is None:
            raise RuntimeError("Fit PCA before reconstructing faces")
        return np.asarray(projected) @ self.components.T + self.mean
