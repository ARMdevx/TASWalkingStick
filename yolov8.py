import cv2
import time
import threading
import queue
import math

from ultralytics import YOLO

# VL53L5CX Python driver
from vl53l5cx.vl53l5cx import VL53L5CX


# ============================================================
# SETTINGS
# ============================================================

CONF_THRESHOLD = 0.75

# Don't say the same thing constantly
SPEECH_COOLDOWN = 3.0

# YOLO processing rate
YOLO_INTERVAL = 0.10       # ~10 detections/sec maximum

# Distance sensor update rate
TOF_INTERVAL = 0.08        # ~12.5 readings/sec

# Only consider objects that are reasonably large in the image
MIN_OBJECT_AREA = 0.015

# Objects we care about most
IMPORTANT_OBJECTS = {
    "person",
    "car",
    "truck",
    "bus",
    "motorcycle",
    "bicycle",
    "dog",
    "chair",
    "bench",
    "backpack",
    "suitcase"
}


# ============================================================
# YOLO
# ============================================================

print("Loading YOLO...")

model = YOLO("yolov8n.pt")

print("YOLO loaded.")


# ============================================================
# SHARED DATA
# ============================================================

latest_frame = None

latest_detection = None

latest_distances = None

frame_lock = threading.Lock()
detection_lock = threading.Lock()
distance_lock = threading.Lock()

running = True


# ============================================================
# SPEECH
# ============================================================

speech_queue = queue.Queue()

last_spoken = {}


def speech_worker():

    import pyttsx3

    engine = pyttsx3.init()

    # Slightly slower than default
    engine.setProperty("rate", 175)

    while running:

        try:
            message = speech_queue.get(timeout=0.2)

        except queue.Empty:
            continue

        if message is None:
            break

        try:
            engine.say(message)
            engine.runAndWait()

        except Exception as e:
            print("Speech error:", e)


speech_thread = threading.Thread(
    target=speech_worker,
    daemon=True
)

speech_thread.start()


def speak(message, key):

    now = time.time()

    if (
        key not in last_spoken
        or now - last_spoken[key] >= SPEECH_COOLDOWN
    ):

        speech_queue.put(message)

        last_spoken[key] = now


# ============================================================
# CAMERA THREAD
# ============================================================

def camera_worker():

    global latest_frame
    global running

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():

        print("ERROR: Could not open webcam.")

        running = False
        return

    # Try to keep latency low
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    while running:

        ret, frame = cap.read()

        if not ret:
            continue

        with frame_lock:

            latest_frame = frame

    cap.release()


# ============================================================
# YOLO THREAD
# ============================================================

def yolo_worker():

    global latest_detection

    last_time = 0

    while running:

        now = time.time()

        if now - last_time < YOLO_INTERVAL:

            time.sleep(0.005)
            continue

        last_time = now

        with frame_lock:

            if latest_frame is None:
                continue

            frame = latest_frame.copy()

        # Smaller image = faster inference
        small_frame = cv2.resize(
            frame,
            (640, 360)
        )

        results = model(
            small_frame,
            verbose=False
        )

        if len(results) == 0:
            continue

        result = results[0]

        detections = []

        frame_height, frame_width = small_frame.shape[:2]

        frame_area = frame_width * frame_height

        for box in result.boxes:

            confidence = float(box.conf[0])

            if confidence < CONF_THRESHOLD:
                continue

            class_id = int(box.cls[0])

            label = model.names[class_id]

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            width = x2 - x1
            height = y2 - y1

            area = width * height

            area_ratio = area / frame_area

            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            # Ignore tiny detections
            if area_ratio < MIN_OBJECT_AREA:
                continue

            # ------------------------------------------------
            # Determine horizontal position
            # ------------------------------------------------

            if center_x < frame_width * 0.35:

                direction = "left"

            elif center_x > frame_width * 0.65:

                direction = "right"

            else:

                direction = "ahead"

            # ------------------------------------------------
            # Determine vertical position
            # ------------------------------------------------

            if center_y < frame_height * 0.35:

                vertical = "upper"

            elif center_y > frame_height * 0.70:

                vertical = "lower"

            else:

                vertical = "middle"

            # ------------------------------------------------
            # Threat score
            # ------------------------------------------------

            score = 0

            # Higher confidence = better
            score += confidence * 5

            # Important objects get priority
            if label in IMPORTANT_OBJECTS:

                score += 10

            # Objects closer to the center are more relevant
            center_distance = abs(
                center_x - frame_width / 2
            ) / (frame_width / 2)

            center_score = 1 - center_distance

            score += center_score * 10

            # Larger objects are generally more relevant
            score += min(area_ratio * 30, 10)

            detections.append({
                "label": label,
                "confidence": confidence,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "center_x": center_x,
                "center_y": center_y,
                "direction": direction,
                "vertical": vertical,
                "area_ratio": area_ratio,
                "score": score
            })

        # ----------------------------------------------------
        # Choose the most threatening object
        # ----------------------------------------------------

        if detections:

            detections.sort(
                key=lambda x: x["score"],
                reverse=True
            )

            best = detections[0]

            with detection_lock:

                latest_detection = best

        else:

            with detection_lock:

                latest_detection = None


# ============================================================
# VL53L5CX THREAD
# ============================================================

def tof_worker():

    global latest_distances

    print("Starting VL53L5CX...")

    try:

        sensor = VL53L5CX()

        print("Initializing VL53L5CX...")

        sensor.init()

        # 8x8 = 64 zones
        sensor.set_resolution_8x8()

        # 8x8 can run up to 15 Hz
        sensor.set_ranging_frequency_hz(15)

        sensor.start_ranging()

        print("VL53L5CX ready.")

    except Exception as e:

        print("VL53L5CX initialization failed:")
        print(e)

        running_local = True

        while running_local and running:
            time.sleep(1)

        return

    while running:

        try:

            if not sensor.data_ready():

                time.sleep(0.005)
                continue

            data = sensor.get_data()

            # distance_mm contains 64 measurements
            distances = list(data.distance_mm)

            with distance_lock:

                latest_distances = distances

        except Exception as e:

            print("ToF error:", e)

            time.sleep(0.1)


# ============================================================
# MAP YOLO POSITION → VL53L5CX ZONES
# ============================================================

def get_object_distance(detection, distances):

    if distances is None:
        return None

    if len(distances) != 64:
        return None

    # YOLO image is 640 x 360
    image_width = 640
    image_height = 360

    cx = detection["center_x"]
    cy = detection["center_y"]

    # Convert image coordinate to 8x8 zone
    zone_x = int(
        (cx / image_width) * 8
    )

    zone_y = int(
        (cy / image_height) * 8
    )

    zone_x = max(
        0,
        min(7, zone_x)
    )

    zone_y = max(
        0,
        min(7, zone_y)
    )

    # --------------------------------------------------------
    # Instead of using ONE zone, use a small neighborhood.
    #
    # This makes the distance more stable when YOLO's box
    # moves slightly from frame to frame.
    # --------------------------------------------------------

    nearby_distances = []

    for dy in [-1, 0, 1]:

        for dx in [-1, 0, 1]:

            x = zone_x + dx
            y = zone_y + dy

            if x < 0 or x >= 8:
                continue

            if y < 0 or y >= 8:
                continue

            index = y * 8 + x

            distance = distances[index]

            # Ignore invalid readings
            if distance is None:
                continue

            if distance <= 0:
                continue

            # VL53L5CX gives mm
            if distance > 4500:
                continue

            nearby_distances.append(distance)

    if not nearby_distances:
        return None

    # Use the closest valid reading in the object region.
    #
    # This is useful because the object may only occupy
    # part of the 3x3 region.
    distance_mm = min(nearby_distances)

    return distance_mm


# ============================================================
# DISTANCE → URGENCY
# ============================================================

def distance_to_message(
    label,
    direction,
    distance_mm
):

    if distance_mm is None:

        if direction == "ahead":

            return f"{label} ahead"

        else:

            return f"{label} on your {direction}"

    distance_m = distance_mm / 1000.0

    # Round nicely
    distance_m = round(
        distance_m,
        1
    )

    # --------------------------------------------------------
    # Very close
    # --------------------------------------------------------

    if distance_m <= 0.7:

        if direction == "ahead":

            return f"Warning. {label} very close ahead."

        else:

            return (
                f"Warning. {label} very close "
                f"on your {direction}."
            )

    # --------------------------------------------------------
    # Medium distance
    # --------------------------------------------------------

    if direction == "ahead":

        return (
            f"{label} ahead, "
            f"{distance_m} meters"
        )

    else:

        return (
            f"{label} on your {direction}, "
            f"{distance_m} meters"
        )


# ============================================================
# DRAW DETECTION
# ============================================================

def draw_detection(frame, detection, distance_mm):

    if detection is None:
        return frame

    x1 = int(detection["x1"])
    y1 = int(detection["y1"])
    x2 = int(detection["x2"])
    y2 = int(detection["y2"])

    label = detection["label"]

    confidence = detection["confidence"]

    direction = detection["direction"]

    # Box
    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    # Label
    text = (
        f"{label} "
        f"{confidence:.2f}"
    )

    if distance_mm is not None:

        text += (
            f" | {distance_mm / 1000:.1f}m"
        )

    cv2.putText(
        frame,
        text,
        (x1, max(25, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2
    )

    return frame


# ============================================================
# START THREADS
# ============================================================

camera_thread = threading.Thread(
    target=camera_worker,
    daemon=True
)

yolo_thread = threading.Thread(
    target=yolo_worker,
    daemon=True
)

tof_thread = threading.Thread(
    target=tof_worker,
    daemon=True
)

camera_thread.start()

yolo_thread.start()

tof_thread.start()


# ============================================================
# MAIN DISPLAY / DECISION LOOP
# ============================================================

print("Smart Cane running.")
print("Press Q to quit.")

last_spoken_detection = None

while running:

    # Get newest camera frame

    with frame_lock:

        if latest_frame is None:

            time.sleep(0.01)
            continue

        display = latest_frame.copy()

    # Get newest YOLO detection

    with detection_lock:

        detection = latest_detection

    # Get newest ToF measurements

    with distance_lock:

        distances = latest_distances

    # --------------------------------------------------------
    # Calculate distance to selected object
    # --------------------------------------------------------

    distance_mm = get_object_distance(
        detection,
        distances
    )

    # --------------------------------------------------------
    # Draw selected object
    # --------------------------------------------------------

    display = draw_detection(
        display,
        detection,
        distance_mm
    )

    # --------------------------------------------------------
    # Decide what to say
    # --------------------------------------------------------

    if detection is not None:

        label = detection["label"]

        direction = detection["direction"]

        confidence = detection["confidence"]

        message = distance_to_message(
            label,
            direction,
            distance_mm
        )

        # Speech key changes when the object/direction changes
        # or when distance crosses an important threshold.

        if distance_mm is not None:

            distance_m = distance_mm / 1000.0

            if distance_m <= 0.7:

                distance_band = "very_close"

            elif distance_m <= 1.5:

                distance_band = "close"

            elif distance_m <= 3.0:

                distance_band = "medium"

            else:

                distance_band = "far"

        else:

            distance_band = "unknown"

        speech_key = (
            f"{label}_"
            f"{direction}_"
            f"{distance_band}"
        )

        speak(
            message,
            speech_key
        )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    cv2.imshow(
        "Smart Cane",
        display
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):

        running = False

        break


# ============================================================
# CLEANUP
# ============================================================

running = False

speech_queue.put(None)

cv2.destroyAllWindows()

print("Smart Cane stopped.")