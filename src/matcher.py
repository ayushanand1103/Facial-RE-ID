"""Train, persist, calibrate, and fuse mathematical face matching scores."""

import json
import hashlib
from pathlib import Path

import numpy as np

from .pca import Eigenfaces
from .mahalanobis import MahalanobisMetric
from .riemannian import covariance_descriptor, spd_distance


class FaceMatcher:
    def __init__(self, n_components=50, alpha=0.5, regularization=0.1):
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must lie in [0, 1]")
        self.alpha = float(alpha)
        self.pca = Eigenfaces(n_components)
        self.metric = MahalanobisMetric(regularization)
        self.threshold = None
        self.calibration = {"status": "not trained"}

    @staticmethod
    def _faces(faces):
        faces = np.asarray(faces, dtype=np.float64)
        if faces.ndim != 3 or faces.shape[1:] != (64, 64) or len(faces) == 0:
            raise ValueError("Expected one or more normalized 64 x 64 faces")
        if not np.isfinite(faces).all() or np.any(faces < 0) or np.any(faces > 1):
            raise ValueError("Face pixels must be finite and in [0, 1]")
        return faces

    def fit(self, faces, labels):
        faces = self._faces(faces)
        labels = np.asarray(labels, dtype=str)
        if labels.shape != (len(faces),) or np.any(labels == "") or np.any(np.char.lower(labels) == "unknown"):
            raise ValueError("Provide one nonempty identity label per face")
        identities, counts = np.unique(labels, return_counts=True)
        if len(identities) < 2 or counts.min() < 3:
            raise ValueError("Enroll at least two identities, with at least three faces each")
        self.labels = labels
        self.training_hashes = [self._fingerprint(face) for face in faces]
        self.pca.fit(faces.reshape(len(faces), -1))
        self.features = self.pca.transform(faces.reshape(len(faces), -1))
        self.metric.fit(self.features, labels)
        self.descriptors = np.array([covariance_descriptor(face) for face in faces])
        # Fixed gallery-derived scales: never min-max normalize across query candidates.
        # Compute within-identity pairs, capping scale estimation at 30 samples per person.
        mahalanobis_pairs, riemannian_pairs = [], []
        for identity in identities:
            indices = np.flatnonzero(labels == identity)[:30]
            for position, index in enumerate(indices[:-1]):
                others = indices[position + 1:]
                mahalanobis_pairs.extend(self.metric.distances(self.features[index], self.features[others]))
                riemannian_pairs.extend(spd_distance(self.descriptors[index], self.descriptors[j]) for j in others)
        self.mahalanobis_scale = self._scale(mahalanobis_pairs)
        self.riemannian_scale = self._scale(riemannian_pairs)
        # Leave-one-out scoring only: PCA/metric still used all training samples.
        # This is a provisional operating point, not held-out accuracy evidence.
        genuine_scores = []
        for index in range(len(faces)):
            distances = self._distances(self.features[index], self.descriptors[index])[2]
            peers = (labels == labels[index]) & (np.arange(len(labels)) != index)
            genuine_scores.append(float(distances[peers].min()))
        self.threshold = max(float(np.quantile(genuine_scores, 0.95)), 1e-6)
        self.calibration = {
            "status": "provisional", "method": "training leave-one-out scores; no refitting",
            "genuine_samples": len(genuine_scores), "unknown_samples": 0,
        }
        return self

    @staticmethod
    def _fingerprint(face):
        return hashlib.sha256(np.asarray(face, dtype=np.float32).tobytes()).hexdigest()

    def _check_holdout(self, faces):
        if any(self._fingerprint(face) in self.training_hashes for face in faces):
            raise ValueError("Validation/test contains an exact training crop; capture separate images")

    @staticmethod
    def _scale(distances):
        values = np.asarray(distances)
        positive = values[values > 1e-10]
        return max(float(np.median(positive)) if positive.size else 1.0, 1e-6)

    def _distances(self, feature, descriptor):
        dm = self.metric.distances(feature, self.features)
        dr = np.array([spd_distance(descriptor, enrolled) for enrolled in self.descriptors])
        fused = self.alpha * dm / self.mahalanobis_scale + (1 - self.alpha) * dr / self.riemannian_scale
        return dm, dr, fused

    def match(self, face, threshold=None):
        if self.threshold is None:
            raise RuntimeError("Train or load a gallery before matching")
        face = self._faces(np.asarray(face)[None])[0]
        feature = self.pca.transform(face.ravel())
        dm, dr, fused = self._distances(feature, covariance_descriptor(face))
        candidates = []
        for identity in np.unique(self.labels):
            indices = np.flatnonzero(self.labels == identity)
            best = int(indices[np.argmin(fused[indices])])
            candidates.append({
                "identity": str(identity), "score": float(fused[best]),
                "mahalanobis": float(dm[best]), "riemannian": float(dr[best]),
            })
        candidates.sort(key=lambda item: item["score"])
        best = candidates[0]
        cutoff = self.threshold if threshold is None else float(threshold)
        if not np.isfinite(cutoff) or cutoff < 0:
            raise ValueError("Threshold must be finite and nonnegative")
        return {
            "identity": best["identity"] if best["score"] <= cutoff else "Unknown",
            "closest_identity": best["identity"], "score": best["score"],
            "threshold": cutoff, "calibration": self.calibration["status"],
            "candidates": candidates,
        }

    def calibrate(self, faces, labels, max_false_accept_rate=0.05):
        """Choose a threshold on a separate validation set with known and unknown people."""
        faces = self._faces(faces)
        self._check_holdout(faces)
        labels = np.asarray(labels, dtype=str)
        if labels.shape != (len(faces),) or not 0 <= max_false_accept_rate <= 1:
            raise ValueError("Invalid calibration labels or false-accept target")
        genuine, unknown = [], []
        for face, label in zip(faces, labels):
            result = self.match(face)
            if label not in self.labels:
                unknown.append(result["score"])
            elif result["closest_identity"] == label:
                genuine.append(result["score"])
        if not genuine or not unknown:
            raise ValueError("Calibration needs correctly matched known faces and unknown identities")
        genuine, unknown = np.array(genuine), np.array(unknown)
        target = float(np.quantile(genuine, 0.95))
        options = np.unique(np.concatenate(([0.0, target], genuine[genuine <= target],
                                            np.nextafter(unknown[unknown <= target], -np.inf))))
        feasible = [float(t) for t in options if t >= 0 and np.mean(unknown <= t) <= max_false_accept_rate]
        if not feasible:
            raise ValueError("No nonnegative threshold satisfies the requested validation false-accept rate")
        threshold = max(feasible)
        # Evaluate all validation samples, including misclassified known identities.
        report = self.evaluate(faces, labels, threshold=threshold)
        self.threshold = threshold
        self.calibration = {
            "status": "held-out", "max_false_accept_rate": max_false_accept_rate,
            "report": report,
        }
        return report

    def evaluate(self, faces, labels, threshold=None):
        faces = self._faces(faces)
        self._check_holdout(faces)
        labels = np.asarray(labels, dtype=str)
        if labels.shape != (len(faces),):
            raise ValueError("Provide one validation label per face")
        known_count = unknown_count = correct_known = correct_unknown = 0
        for face, label in zip(faces, labels):
            result = self.match(face, threshold)
            if label in self.labels:
                known_count += 1
                correct_known += result["identity"] == label
            else:
                unknown_count += 1
                correct_unknown += result["identity"] == "Unknown"
        return {
            "samples": len(faces), "known_samples": known_count, "unknown_samples": unknown_count,
            "known_identification_rate": correct_known / known_count if known_count else None,
            "unknown_rejection_rate": correct_unknown / unknown_count if unknown_count else None,
            "false_accept_rate": 1 - correct_unknown / unknown_count if unknown_count else None,
        }

    def save(self, path):
        if self.threshold is None:
            raise RuntimeError("Train a gallery before saving")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = json.dumps({
            "version": 1, "alpha": self.alpha, "threshold": self.threshold,
            "mahalanobis_scale": self.mahalanobis_scale, "riemannian_scale": self.riemannian_scale,
            "regularization": self.metric.regularization, "calibration": self.calibration,
            "training_hashes": self.training_hashes,
        })
        # No pickle: arrays and a JSON string only. Replace atomically after writing.
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("wb") as file:
            np.savez_compressed(file, metadata=metadata, labels=self.labels, features=self.features,
                                descriptors=self.descriptors, mean=self.pca.mean,
                                components=self.pca.components, eigenvalues=self.pca.eigenvalues,
                                variance_ratio=self.pca.explained_variance_ratio,
                                covariance=self.metric.covariance)
        temporary.replace(path)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as saved:
            metadata = json.loads(str(saved["metadata"]))
            if metadata["version"] != 1:
                raise ValueError("Unsupported gallery version; retrain the gallery")
            model = cls(saved["components"].shape[1], metadata["alpha"], metadata["regularization"])
            for name in ("labels", "features", "descriptors"):
                setattr(model, name, saved[name].copy())
            for name, key in (("mean", "mean"), ("components", "components"),
                              ("eigenvalues", "eigenvalues"), ("explained_variance_ratio", "variance_ratio")):
                setattr(model.pca, name, saved[key].copy())
            model.metric.covariance = saved["covariance"].copy()
            model.metric.cholesky = np.linalg.cholesky(model.metric.covariance)
            for name in ("threshold", "mahalanobis_scale", "riemannian_scale", "calibration", "training_hashes"):
                setattr(model, name, metadata[name])
        return model
