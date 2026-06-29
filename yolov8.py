import cv2
import time
import threading
import queue
from ultralytics import YOLO
import pythoncom
import win32com.client

# =========================
# MODEL
# =========================
model = YOLO("yolov8n.pt")

# =========================
# SPEECH ENGINE (QUEUE BASED)
# =========================
speech_queue = queue.Queue()

speaker = None

def speech_worker():
    global speaker
    pythoncom.CoInitialize()
    speaker = win32com.client.Dispatch("SAPI.SpVoice")
    speaker.Rate = 1

    while True:
        text = speech_queue.get()
        if text is None:
            break
        speaker.Speak(text)

threading.Thread(target=speech_worker, daemon=True).start()

def speak(text):
    speech_queue.put(text)

# =========================
# SETTINGS
# =========================
CONF_THRESHOLD = 0.75
COOLDOWN = 2.5
FRAME_SKIP = 2

IMPORTANT_OBJECTS = {
    "person", "car", "truck", "bus",
    "bicycle", "motorcycle", "chair",
    "bench", "dog"
}

DANGER_OBJECTS = IMPORTANT_OBJECTS

last_spoken = {}
latest_frame = None
latest_results = None
lock = threading.Lock()

# =========================
# YOLO THREAD (runs separately)
# =========================
def yolo_loop():
    global latest_frame, latest_results

    frame_id = 0

    while True:
        if latest_frame is None:
            continue

        frame_id += 1

        if frame_id % FRAME_SKIP != 0:
            continue

        frame = cv2.resize(latest_frame, (640, 360))
        results = model(frame, verbose=False)

        with lock:
            latest_results = results
            latest_frame_shape = frame.shape

# =========================
# START YOLO THREAD
# =========================
threading.Thread(target=yolo_loop, daemon=True).start()

# =========================
# CAMERA
# =========================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not found")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    latest_frame = frame.copy()

    display = frame.copy()
    now = time.time()

    with lock:
        results = latest_results

    if results is not None:

        h, w = 360, 640
        frame_area = h * w

        best_event = None
        best_score = 0

        for box in results[0].boxes:

            conf = float(box.conf[0])
            if conf < CONF_THRESHOLD:
                continue

            cls = int(box.cls[0])
            label = model.names[cls]

            x1, y1, x2, y2 = box.xyxy[0]
            area = (x2 - x1) * (y2 - y1)
            cx = (x1 + x2) / 2

            in_center = (w * 0.35 < cx < w * 0.65)
            size_ratio = area / frame_area

            score = 0
            if in_center:
                score += 5
            if label in DANGER_OBJECTS:
                score += 4
            if size_ratio > 0.15:
                score += 6

            unknown_large = (label not in IMPORTANT_OBJECTS) and (size_ratio > 0.15)
            if unknown_large:
                score += 8

            if score > best_score:
                best_score = score

                if unknown_large:
                    msg = "Obstacle ahead"
                    key = "obstacle"
                else:
                    if in_center:
                        msg = f"{label} ahead"
                    else:
                        direction = "left" if cx < w / 2 else "right"
                        msg = f"{label} on your {direction}"

                    key = f"{label}_{msg}"

                best_event = (key, msg)

        if best_event:
            key, msg = best_event

            if key not in last_spoken or now - last_spoken[key] > COOLDOWN:
                speak(msg)
                last_spoken[key] = now

    cv2.imshow("Smart Cane (Real-Time)", display)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
speech_queue.put(None)