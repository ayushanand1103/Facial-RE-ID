import cv2


def start_cam(camera_index=0):
    camera = None
    frame = None
    # Try the Windows DirectShow backend, then OpenCV's default backend.
    for backend in (cv2.CAP_DSHOW, cv2.CAP_ANY):
        print(f"Opening camera {camera_index} with backend {backend}...", flush=True)
        candidate = cv2.VideoCapture(camera_index, backend)
        if candidate.isOpened():
            success, first_frame = candidate.read()
            if success and first_frame is not None:
                camera, frame = candidate, first_frame
                break
        candidate.release()

    if camera is None:
        raise RuntimeError(
            "Could not capture from the camera. Close other camera apps, "
            "check Windows Settings > Privacy & security > Camera and allow "
            "desktop apps to access the camera, or try start_cam(camera_index=1)."
        )

    try:
        print("Camera opened. Click the video window and press Q to quit.")
        while True:
            cv2.imshow("Facial Re-ID | Camera", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            success, frame = camera.read()
            if not success or frame is None:
                raise RuntimeError("Camera opened, but stopped delivering frames.")
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    start_cam()
