"""Numerical and workflow checks; synthetic data does not establish face accuracy."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.mahalanobis import MahalanobisMetric
from src.matcher import FaceMatcher
from src.pca import Eigenfaces
from src.preprocessing import FacePreprocessor
from src.riemannian import covariance_descriptor, spd_distance
from src.workflow import read_dataset


def synthetic_faces(seed=42, count=5):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:64, 0:64] / 63
    bases = [0.2 + 0.35 * x + 0.1 * np.sin(8 * y),
             0.45 + 0.25 * y + 0.1 * np.cos(9 * x)]
    faces, labels = [], []
    for index, base in enumerate(bases):
        for _ in range(count):
            faces.append(np.clip(base + rng.normal(0, 0.01, base.shape), 0, 1))
            labels.append(f"person{index}")
    return np.array(faces), np.array(labels)


class MathTests(unittest.TestCase):
    def test_preprocessing_bounds_normalization_and_order(self):
        frame = np.array([[0, 100], [200, 255]], dtype=np.uint8)
        result = FacePreprocessor((2, 2)).process(frame, (-1, -2, 5, 5))
        np.testing.assert_allclose(result["vector"], np.array([0, 100, 200, 255]) / 255)
        self.assertEqual(result["vector"].dtype, np.float32)
        with self.assertRaises(ValueError):
            FacePreprocessor().process(frame, (4, 4, 5, 5))
        with self.assertRaises(ValueError):
            FacePreprocessor().process(frame.astype(float), (0, 0, 2, 2))

    def test_pca_matches_covariance_eigenvalues_and_reconstructs(self):
        samples = np.random.default_rng(2).normal(size=(12, 5))
        pca = Eigenfaces(5).fit(samples)
        expected = np.linalg.eigvalsh(np.cov(samples, rowvar=False))[::-1]
        np.testing.assert_allclose(pca.eigenvalues, expected, rtol=1e-10)
        np.testing.assert_allclose(pca.components.T @ pca.components, np.eye(5), atol=1e-12)
        np.testing.assert_allclose(pca.transform(samples).mean(axis=0), 0, atol=1e-12)
        np.testing.assert_allclose(pca.inverse_transform(pca.transform(samples)), samples, atol=1e-12)
        with self.assertRaises(ValueError):
            Eigenfaces().fit(np.ones((3, 5)))

    def test_mahalanobis_equals_quadratic_form(self):
        features = np.array([[0, 1], [1, 2], [2, 1], [6, 7], [7, 8], [8, 7]], dtype=float)
        metric = MahalanobisMetric().fit(features, ["a"] * 3 + ["b"] * 3)
        query = np.array([1, 1])
        delta = features - query
        expected = np.sqrt(np.einsum("ni,ij,nj->n", delta, np.linalg.inv(metric.covariance), delta))
        np.testing.assert_allclose(metric.distances(query, features), expected)
        self.assertGreater(np.linalg.eigvalsh(metric.covariance).min(), 0)

    def test_spd_properties_and_affine_invariance(self):
        a = np.diag([1.0, 2.0, 3.0])
        b = np.diag([2.0, 3.0, 4.0])
        np.testing.assert_allclose(spd_distance(a, b), np.linalg.norm(np.log([2, 1.5, 4/3])))
        self.assertAlmostEqual(spd_distance(a, a), 0, places=12)
        self.assertAlmostEqual(spd_distance(a, b), spd_distance(b, a), places=12)
        transform = np.array([[2., 1, 0], [0, 1, 0.1], [1, 0, 2]])
        self.assertAlmostEqual(spd_distance(a, b),
                               spd_distance(transform @ a @ transform.T, transform @ b @ transform.T), places=10)
        descriptor = covariance_descriptor(np.zeros((64, 64)))
        self.assertGreater(np.linalg.eigvalsh(descriptor).min(), 0)
        with self.assertRaises(ValueError):
            spd_distance(a, np.diag([-1., 2., 3.]))

    def test_matching_persistence_calibration_and_unknown(self):
        faces, labels = synthetic_faces()
        matcher = FaceMatcher(n_components=5).fit(faces, labels)
        for face, label in zip(faces, labels):
            self.assertEqual(matcher.match(face)["identity"], label)
        with self.assertRaises(ValueError):
            matcher.calibrate(faces, labels)
        validation, validation_labels = synthetic_faces(seed=5, count=2)
        unknown = np.zeros((1, 64, 64))
        validation = np.concatenate([validation, unknown])
        validation_labels = np.concatenate([validation_labels, ["outsider"]])
        report = matcher.calibrate(validation, validation_labels, max_false_accept_rate=0)
        self.assertEqual(report["false_accept_rate"], 0)
        self.assertEqual(matcher.match(unknown[0])["identity"], "Unknown")
        self.assertEqual(matcher.calibration["status"], "held-out")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gallery.npz"
            matcher.save(path)
            loaded = FaceMatcher.load(path)
            before, after = matcher.match(validation[0]), loaded.match(validation[0])
            self.assertEqual(before["identity"], after["identity"])
            self.assertEqual(before["calibration"], after["calibration"])
            self.assertAlmostEqual(before["score"], after["score"], places=12)
            for expected, actual in zip(before["candidates"], after["candidates"]):
                self.assertEqual(expected["identity"], actual["identity"])
                for key in ("score", "mahalanobis", "riemannian"):
                    self.assertAlmostEqual(expected[key], actual[key], places=12)

    def test_fixed_normalization_and_alpha_endpoints(self):
        faces, labels = synthetic_faces(count=3)
        for alpha in (0, 1):
            matcher = FaceMatcher(n_components=3, alpha=alpha).fit(faces, labels)
            dm, dr, fused = matcher._distances(matcher.features[0], matcher.descriptors[0])
            expected = dr / matcher.riemannian_scale if alpha == 0 else dm / matcher.mahalanobis_scale
            np.testing.assert_allclose(fused, expected)
        with self.assertRaises(ValueError):
            FaceMatcher(alpha=2)
        with self.assertRaises(ValueError):
            FaceMatcher().fit(faces[:2], labels[:2])

    def test_single_identity_gallery(self):
        faces, labels = synthetic_faces(count=3)
        matcher = FaceMatcher(n_components=2).fit(faces[:3], labels[:3])
        self.assertEqual(matcher.match(faces[0])["identity"], labels[0])
        self.assertEqual(matcher.match(faces[0], threshold=0)["closest_identity"], labels[0])
        self.assertEqual(matcher.match(np.zeros((64, 64)), threshold=0)["identity"], "Unknown")

    def test_png_dataset_loading(self):
        import cv2
        with tempfile.TemporaryDirectory() as directory:
            person = Path(directory) / "alice"
            person.mkdir()
            cv2.imwrite(str(person / "face.png"), np.full((64, 64), 128, dtype=np.uint8))
            faces, labels = read_dataset(directory)
            self.assertEqual(faces.shape, (1, 64, 64))
            self.assertEqual(labels.tolist(), ["alice"])
            np.testing.assert_allclose(faces, 128 / 255)


if __name__ == "__main__":
    unittest.main()
