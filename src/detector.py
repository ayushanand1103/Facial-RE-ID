
from pathlib import Path

if __package__:
    from .preprocessing import FacePreprocessor
    from .camera import open_camera
else:
    from preprocessing import FacePreprocessor
    from camera import open_camera

import cv2
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "yolov8n-face.pt"


class FaceDetector:
    """Load face YOLO once and return clipped bounding boxes with confidence."""

    def __init__(self, model_path=MODEL_PATH, confidence=0.5, device=None):
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"Face model not found: {model_path}")
        if not 0 < confidence <= 1:
            raise ValueError("Detection confidence must lie in (0, 1]")
        self.device = device if device is not None else (0 if torch.cuda.is_available() else "cpu")
        self.confidence = confidence
        self.model = YOLO(str(model_path))
        if len(self.model.names) != 1 or "face" not in str(self.model.names).lower():
            raise ValueError("Use a face-specific checkpoint, not a standard object model")

    def detect(self, frame):
        results = self.model.predict(source=frame, conf=self.confidence,
                                     device=self.device, verbose=False)
        height, width = frame.shape[:2]
        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                bbox = (max(0, min(int(x1), width)), max(0, min(int(y1), height)),
                        max(0, min(int(x2), width)), max(0, min(int(y2), height)))
                if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                    detections.append({"bbox": bbox, "confidence": float(box.conf[0])})
        return detections


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Face model not found: {MODEL_PATH}"
        )

    device = 0 if torch.cuda.is_available() else "cpu"

    model = YOLO(str(MODEL_PATH))
    preprocessor = FacePreprocessor()
    printed_example = False
    camera, _ = open_camera(0)

    try:
        while True:
            success, frame = camera.read()

            if not success:
                break

            results = model.predict(
                source=frame,
                conf=0.5,
                device=device,
                verbose=False
            )

            # Draw on a copy so annotations cannot contaminate any face crop.
            display_frame = frame.copy()

            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = (
                        box.xyxy[0]
                        .cpu()
                        .numpy()
                        .astype(int)
                    )

                    confidence = float(box.conf[0])

                    try:
                        processed = preprocessor.process(frame, (x1, y1, x2, y2))
                    except ValueError as error:
                        print(f"Skipping invalid face crop: {error}")
                        continue

                    face_vector = processed["vector"]
                    if not printed_example:
                        print("Face matrix:", processed["normalized"].shape)
                        print("Face vector:", face_vector.shape)
                        print("First 10 values:", face_vector[:10])
                        printed_example = True

                    cv2.imshow("Face | 64 x 64 normalized", processed["normalized"])

                    cv2.rectangle(
                        display_frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        display_frame,
                        f"Face: {confidence:.2f}",
                        (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )

            cv2.imshow("Facial Re-ID | YOLO", display_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
