
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "yolov8n-face.pt"

def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Face model not found: {MODEL_PATH}"
        )

    device = 0 if torch.cuda.is_available() else "cpu"

    model = YOLO(str(MODEL_PATH))
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError("Could not open webcam")

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

            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = (
                        box.xyxy[0]
                        .cpu()
                        .numpy()
                        .astype(int)
                    )

                    confidence = float(box.conf[0])

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"Face: {confidence:.2f}",
                        (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )

            cv2.imshow("Facial Re-ID | YOLO", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
