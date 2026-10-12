"""Exercise the CLI with real preprocessing/math and controlled face detections."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from src.matcher import FaceMatcher
from src.workflow import main


class WorkflowTests(unittest.TestCase):
    def test_image_enrollment_training_recognition_and_evaluation(self):
        rng = np.random.default_rng(12)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data, gallery = root / "data", root / "gallery.npz"
            images = {}
            for identity, base in (("alice", 70), ("bob", 170)):
                images[identity] = []
                for index in range(4):
                    frame = np.clip(base + rng.normal(0, 5, (64, 64, 3)), 0, 255).astype(np.uint8)
                    path = root / f"{identity}{index}.png"
                    cv2.imwrite(str(path), frame)
                    images[identity].append(path)
            with patch("src.workflow.FaceDetector") as detector, contextlib.redirect_stdout(io.StringIO()) as output:
                detector.return_value.detect.return_value = [{"bbox": (0, 0, 64, 64), "confidence": 0.99}]
                for identity, paths in images.items():
                    main(["enroll", "--name", identity, "--data", str(data), "--images", *map(str, paths[:3])])
                    main(["enroll", "--name", identity, "--data", str(data), "--split", "test",
                          "--images", str(paths[3])])
                main(["train", "--data", str(data), "--gallery", str(gallery), "--components", "3"])
                self.assertTrue(gallery.exists())
                loaded = FaceMatcher.load(gallery)
                self.assertEqual(set(loaded.labels), {"alice", "bob"})
                main(["identify", "--gallery", str(gallery), "--image", str(images["alice"][0])])
                main(["evaluate", "--data", str(data), "--gallery", str(gallery)])
                self.assertIn('"identity": "alice"', output.getvalue())
                self.assertIn('"known_samples": 2', output.getvalue())

    def test_enrollment_refuses_multiple_faces(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "image.png"
            cv2.imwrite(str(path), np.zeros((64, 64, 3), dtype=np.uint8))
            with patch("src.workflow.FaceDetector") as detector, contextlib.redirect_stdout(io.StringIO()):
                detector.return_value.detect.return_value = [
                    {"bbox": (0, 0, 32, 32)}, {"bbox": (32, 32, 64, 64)}]
                with self.assertRaises(SystemExit) as error:
                    main(["enroll", "--name", "alice", "--data", str(Path(temporary) / "data"),
                          "--images", str(path)])
                self.assertEqual(error.exception.code, 1)
                self.assertFalse((Path(temporary) / "data" / "train" / "alice").exists())


if __name__ == "__main__":
    unittest.main()
