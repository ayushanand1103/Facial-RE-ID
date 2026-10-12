"""CLI workflow for enrollment, training, calibration, and recognition."""

import argparse
import json
import re
import uuid
from pathlib import Path

import cv2
import numpy as np

from .camera import open_camera, start_cam
from .detector import FaceDetector, ROOT, MODEL_PATH
from .matcher import FaceMatcher
from .preprocessing import FacePreprocessor

DATA_PATH = ROOT / "data"
GALLERY_PATH = ROOT / "models" / "gallery.npz"


def read_image(path):
    frame = cv2.imread(str(path))
    if frame is None:
        raise ValueError(f"Cannot read image: {path}")
    return frame


def read_dataset(directory):
    faces, labels = [], []
    for path in sorted(Path(directory).glob("*/*.png")):
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None or image.shape != (64, 64):
            raise ValueError(f"Expected a readable 64 x 64 enrollment PNG: {path}")
        faces.append(image.astype(np.float32) / 255.0)
        labels.append(path.parent.name)
    if not faces:
        raise ValueError(f"No enrolled faces found in {directory}; run enroll first")
    return np.array(faces), np.array(labels)


def save_face(processed, directory):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{uuid.uuid4().hex}.png"
    if not cv2.imwrite(str(path), processed["resized"]):
        raise OSError(f"Could not save face: {path}")
    print(f"Saved {path}", flush=True)


def draw_box(frame, bbox, label, color=(0, 255, 0)):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


def enroll(args):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", args.name) or args.name.lower() == "unknown":
        raise ValueError("Use a name of 1–64 letters/digits/_/-; 'Unknown' is reserved")
    if args.samples < 1:
        raise ValueError("samples must be positive")
    directory = args.data / args.split / args.name
    detector = FaceDetector(args.model, args.confidence, args.device)
    preprocessor = FacePreprocessor()
    saved = 0
    if args.images:
        for path in args.images:
            frame = read_image(path)
            detections = detector.detect(frame)
            if len(detections) != 1:
                print(f"Skipped {path}: expected exactly one face, found {len(detections)}")
                continue
            save_face(preprocessor.process(frame, detections[0]["bbox"]), directory)
            saved += 1
        if not saved:
            raise ValueError("No images contained exactly one detectable face")
        print(f"Enrolled {saved} faces for {args.name} in {args.split}")
        return

    camera, frame = open_camera(args.camera)
    print("Press S to save a face, Q to finish. Vary expression and angle slightly.", flush=True)
    try:
        while saved < args.samples:
            detections = detector.detect(frame)
            display = frame.copy()
            for detection in detections:
                draw_box(display, detection["bbox"], args.name)
            cv2.putText(display, f"Saved {saved}/{args.samples} | S: save | Q: quit", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow("Enrollment", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                if len(detections) != 1:
                    print("Save requires exactly one visible face.", flush=True)
                else:
                    save_face(preprocessor.process(frame, detections[0]["bbox"]), directory)
                    saved += 1
            success, frame = camera.read()
            if not success or frame is None:
                raise RuntimeError("Camera stopped delivering frames")
    finally:
        camera.release()
        cv2.destroyAllWindows()
    print(f"Enrolled {saved} new faces for {args.name} in {args.split}")


def train(args):
    faces, labels = read_dataset(args.data / "train")
    matcher = FaceMatcher(args.components, args.alpha, args.regularization).fit(faces, labels)
    matcher.save(args.gallery)
    print(json.dumps({"gallery": str(args.gallery), "samples": len(faces),
                      "identities": np.unique(labels).tolist(),
                      "components": matcher.pca.components.shape[1],
                      "explained_variance": float(matcher.pca.explained_variance_ratio.sum()),
                      "threshold": matcher.threshold, "calibration": matcher.calibration}, indent=2))


def identify(args):
    matcher = FaceMatcher.load(args.gallery)
    detector = FaceDetector(args.model, args.confidence, args.device)
    preprocessor = FacePreprocessor()
    print(f"Unknown threshold status: {matcher.calibration['status']}", flush=True)

    def recognize(frame):
        results = []
        for detection in detector.detect(frame):
            processed = preprocessor.process(frame, detection["bbox"])
            result = matcher.match(processed["normalized"], args.threshold)
            result["bbox"] = detection["bbox"]
            results.append(result)
        return results

    if args.image:
        print(json.dumps(recognize(read_image(args.image)), indent=2))
        return
    camera, frame = open_camera(args.camera)
    print("Click the recognition window and press Q to quit.", flush=True)
    try:
        while True:
            results = recognize(frame)
            display = frame.copy()
            for result in results:
                color = (0, 0, 255) if result["identity"] == "Unknown" else (0, 255, 0)
                draw_box(display, result["bbox"], f"{result['identity']} | D={result['score']:.3f}", color)
            cv2.imshow("Facial Re-ID | Recognition", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            success, frame = camera.read()
            if not success or frame is None:
                raise RuntimeError("Camera stopped delivering frames")
    finally:
        camera.release()
        cv2.destroyAllWindows()


def assess(args):
    matcher = FaceMatcher.load(args.gallery)
    faces, labels = read_dataset(args.data / args.split)
    if args.command == "calibrate":
        report = matcher.calibrate(faces, labels, args.max_false_accept_rate)
        matcher.save(args.gallery)
    else:
        report = matcher.evaluate(faces, labels)
    print(json.dumps({"threshold": matcher.threshold, "calibration": matcher.calibration,
                      "evaluation": report}, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Mathematical face re-identification")
    commands = parser.add_subparsers(dest="command", required=True)

    def storage_options(command):
        command.add_argument("--data", type=Path, default=DATA_PATH)
        command.add_argument("--gallery", type=Path, default=GALLERY_PATH)

    def detector_options(command):
        command.add_argument("--model", type=Path, default=MODEL_PATH)
        command.add_argument("--confidence", type=float, default=0.5)
        command.add_argument("--device", help="cpu or CUDA device index, e.g. 0")
        command.add_argument("--camera", type=int, default=0)

    camera_command = commands.add_parser("camera", help="Preview the webcam")
    camera_command.add_argument("--camera", type=int, default=0)
    camera_command.set_defaults(handler=lambda args: start_cam(args.camera))

    enroll_command = commands.add_parser("enroll", help="Save detected face crops")
    storage_options(enroll_command)
    detector_options(enroll_command)
    enroll_command.add_argument("--name", required=True)
    enroll_command.add_argument("--samples", type=int, default=10)
    enroll_command.add_argument("--split", choices=("train", "validation", "test"), default="train")
    enroll_command.add_argument("--images", type=Path, nargs="+", help="Use image files instead of webcam")
    enroll_command.set_defaults(handler=enroll)

    train_command = commands.add_parser("train", help="Fit eigenfaces and matching gallery")
    storage_options(train_command)
    train_command.add_argument("--components", type=int, default=50)
    train_command.add_argument("--alpha", type=float, default=0.5)
    train_command.add_argument("--regularization", type=float, default=0.1)
    train_command.set_defaults(handler=train)

    identify_command = commands.add_parser("identify", help="Recognize webcam or image faces")
    storage_options(identify_command)
    detector_options(identify_command)
    identify_command.add_argument("--image", type=Path)
    identify_command.add_argument("--threshold", type=float, help="Override the stored Unknown threshold")
    identify_command.set_defaults(handler=identify)

    for name in ("calibrate", "evaluate"):
        command = commands.add_parser(name, help="Use separately captured validation or test faces")
        storage_options(command)
        command.add_argument("--split", choices=("validation", "test"),
                             default="validation" if name == "calibrate" else "test")
        if name == "calibrate":
            command.add_argument("--max-false-accept-rate", type=float, default=0.05)
        command.set_defaults(handler=assess)

    args = parser.parse_args(argv)
    try:
        args.handler(args)
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(1, f"Error: {error}\n")
