"""Spatial face covariance descriptors and affine-invariant SPD distance."""

import numpy as np
from scipy.linalg import eigh


def covariance_descriptor(face, regularization=1e-3):
    """Covariance over pixels of [x, y, intensity, dI/dx, dI/dy]."""
    face = np.asarray(face, dtype=np.float64)
    if face.ndim != 2 or min(face.shape) < 2 or not np.isfinite(face).all():
        raise ValueError("Descriptor requires a finite 2D face with both sides >= 2")
    if np.any(face < 0) or np.any(face > 1) or regularization <= 0:
        raise ValueError("Face must be normalized to [0, 1]; regularization must be positive")
    height, width = face.shape
    y, x = np.meshgrid(np.linspace(0, 1, height), np.linspace(0, 1, width), indexing="ij")
    dy, dx = np.gradient(face)
    observations = np.column_stack([x.ravel(), y.ravel(), face.ravel(), dx.ravel(), dy.ravel()])
    centered = observations - observations.mean(axis=0)
    covariance = centered.T @ centered / (len(observations) - 1)
    # A ridge makes even flat or degenerate crops strictly positive definite.
    scale = max(float(np.trace(covariance) / covariance.shape[0]), 1e-6)
    return covariance + regularization * scale * np.eye(covariance.shape[0])


def spd_distance(first, second):
    """d(A,B) = sqrt(sum(log(lambda_j)^2)), with B v = lambda A v."""
    first, second = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    if first.ndim != 2 or first.shape[0] != first.shape[1] or first.shape != second.shape:
        raise ValueError("SPD matrices must be square and have matching dimensions")
    if not np.allclose(first, first.T) or not np.allclose(second, second.T):
        raise ValueError("SPD matrices must be symmetric")
    # scipy.linalg.eigh checks the second argument is positive definite.
    eigenvalues = eigh(second, first, eigvals_only=True)
    if np.any(eigenvalues <= 0) or not np.isfinite(eigenvalues).all():
        raise ValueError("Both descriptor matrices must be positive definite")
    return float(np.linalg.norm(np.log(eigenvalues)))
