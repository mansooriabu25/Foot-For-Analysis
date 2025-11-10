import pandas as pd
import cv2
import time
import threading


VIDEO_SRC = 0
OUTPUT_WINDOW_NAME = "Face Attendance (OpenCV) - press 'q' or Stop in UI"

# -------------------------- Camera thread --------------------------
def camera_thread_fn(engine , stop_event: threading.Event):
    cap = cv2.VideoCapture(VIDEO_SRC)
    try:
        cap.set(cv2.CAP_PROP_FPS, 60)
    except Exception:
        pass

    if not cap.isOpened():
        print("ERROR: could not open camera")
        stop_event.set()
        return
    cv2.namedWindow(OUTPUT_WINDOW_NAME, cv2.WINDOW_NORMAL)
    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                print("Frame read failed")
                break
            annotated, events = engine.step_on_frame(frame)
            cv2.imshow(OUTPUT_WINDOW_NAME, annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
                break
    finally:
        try:
            cap.release()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        stop_event.set()
