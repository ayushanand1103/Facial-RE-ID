
import cv2
import numpy as np


class FacePreprocessor:
    """Convert uint8 webcam face crops into normalized, row-major vectors."""

    def __init__(self, image_size=(64, 64)):
        self.image_size = image_size

    def crop_face(self, frame, bbox):
        """
        Extract a face using YOLO bounding box coordinates.
        bbox = (x1, y1, x2, y2)
        """
        x1, y1, x2, y2 = map(int, bbox)

        height, width = frame.shape[:2]

        # Ensure bounding box stays inside image
        x1 = max(0, min(x1, width))
        x2 = max(0, min(x2, width))
        y1 = max(0, min(y1, height))
        y2 = max(0, min(y2, height))

        if x2 <= x1 or y2 <= y1:
            raise ValueError("Invalid face bounding box")

        return frame[y1:y2, x1:x2]

    def to_grayscale(self, face):
        """Convert BGR image into grayscale."""
        if face.ndim == 2:
            return face

        return cv2.cvtColor(
            face, cv2.COLOR_BGR2GRAY
        )

    def resize_face(self, face):
        """Resize face to 64 x 64."""
        return cv2.resize(
            face,
            self.image_size,
            interpolation=cv2.INTER_AREA
        )

    def normalize(self, face):
        """Convert pixel values from [0,255] to [0,1]."""
        return face.astype(np.float32) / 255.0

    def vectorize(self, face):
        """Flatten rows in order; a 64 x 64 matrix yields shape (4096,)."""
        return face.reshape(-1)

    def process(self, frame, bbox):
        """Return intermediate images and a float32 vector in [0, 1]."""

        if frame.dtype != np.uint8:
            raise ValueError("Expected a uint8 frame with pixel values in [0, 255]")

        cropped = self.crop_face(frame, bbox)

        gray = self.to_grayscale(cropped)

        resized = self.resize_face(gray)

        normalized = self.normalize(resized)

        vector = self.vectorize(normalized)

        return {
            "cropped": cropped,
            "grayscale": gray,
            "resized": resized,
            "normalized": normalized,
            "vector": vector
        }
