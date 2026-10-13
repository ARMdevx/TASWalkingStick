import cv2
import time
import threading
import queue
import statistics
import os

from vl53l5cx.vl53l5cx import VL53L5CX


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "yolov8n.onnx"

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

YOLO_SIZE = 320

# Run YOLO every 3rd frame.
DETECTION_INTERVAL = 3

CONFIDENCE_THRESHOLD = 0.60

# ------------------------------------------------------------
# SPEECH SETTINGS
# ------------------------------------------------------------

# Normal announcements must be separated by this much time.
MIN_ANNOUNCEMENT_GAP = 3.0

# Same object must get at least this much closer before
# another normal announcement.
DISTANCE_CHANGE_MM = 500

# If something gets this close, allow an urgent warning.
URGENT_DISTANCE_MM = 800

# Don't announce objects beyond this distance.
MAX_DISTANCE_MM = 3000


# ============================================================
# OBJECT PRIORITIES
# ============================================================

IMPORTANT_OBJECTS = {
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "train",
    "boat",
    "dog",
    "cat",
    "horse",
    "cow",
    "sheep"
}


PRIORITY = {
    "person": 100,
    "car": 100,
    "truck": 100,
    "bus": 100,
    "train": 100,

    "motorcycle": 90,
    "bicycle": 80,

    "dog": 60,
    "horse": 60,
    "cow": 60,
    "sheep": 50,

    "boat": 40,
    "cat": 30
}


# ============================================================
# COCO CLASS NAMES
# ============================================================

CLASS_NAMES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush"
]


# ============================================================
# GLOBAL VARIABLES
# ============================================================

running = True

latest_frame = None
frame_lock = threading.Lock()

latest_distances = None
distance_lock = threading.Lock()

speech_queue = queue.Queue()


# ============================================================
# ANNOUNCEMENT STATE
# ============================================================

last_announced_label = None
last_announced_direction = None
last_announced_distance = None
last_announcement_time = 0

# Used to prevent repeated urgent warnings.
last_urgent_time = 0


# ============================================================
# SPEECH
# ============================================================

def speech_worker():

    while running:

        try:
            message = speech_queue.get(timeout=0.1)

        except queue.Empty:
            continue

        if message is None:
            break

        try:

            # We already confirmed this exact method
            # produces clean sound through the A/V jack.

            safe_message = message.replace('"', "'")

            command = (
                "espeak --stdout "
                + '"' + safe_message + '"'
                + " | aplay -D hw:Headphones -q"
            )

            os.system(command)

        except Exception as e:

            print("TTS ERROR:", e)

        speech_queue.task_done()


def speak(message):

    # Don't build up a huge backlog of speech.
    if speech_queue.empty():

        speech_queue.put(message)


# ============================================================
# CAMERA
# ============================================================

def camera_worker():

    global latest_frame

    camera = cv2.VideoCapture(0)

    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        CAMERA_WIDTH
    )

    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        CAMERA_HEIGHT
    )

    camera.set(
        cv2.CAP_PROP_BUFFERSIZE,
        1
    )

    if not camera.isOpened():

        print("CAMERA ERROR: Could not open USB camera.")

        return

    print("Camera: READY")

    while running:

        success, frame = camera.read()

        if not success:

            time.sleep(0.05)
            continue

        with frame_lock:

            latest_frame = frame

    camera.release()


# ============================================================
# VL53L5CX
#
# THIS IS THE SAME INITIALIZATION SEQUENCE
# FROM YOUR WORKING SENSOR TEST.

# ============================================================

def tof_worker():

    global latest_distances

    sensor = None

    try:

        print("Starting VL53L5CX...")

        sensor = VL53L5CX()

        print("Initializing sensor....")

        sensor.init()

        print("Setting up 8x8 res")

        sensor.set_resolution(64)

        print("Start")

        sensor.start_ranging()

        print("VL53L5CX: READY")

        while running:

            if sensor.check_data_ready():

                data = sensor.get_ranging_data()

                distances = list(data.distance_mm)

                with distance_lock:

                    latest_distances = distances

            time.sleep(0.01)

    except Exception as e:

        print("VL53L5CX ERROR:")
        print(repr(e))

    finally:

        if sensor is not None:

            try:
                sensor.stop_ranging()

            except Exception:
                pass


# ============================================================
# DISTANCE
# ============================================================

def get_object_distance(
    distances,
    x1,
    y1,
    x2,
    y2,
    frame_width,
    frame_height
):

    if distances is None:
        return None

    if len(distances) < 64:
        return None

    left = int(
        (x1 / frame_width) * 8
    )

    right = int(
        (x2 / frame_width) * 8
    )

    top = int(
        (y1 / frame_height) * 8
    )

    bottom = int(
        (y2 / frame_height) * 8
    )

    left = max(0, min(7, left))
    right = max(0, min(7, right))

    top = max(0, min(7, top))
    bottom = max(0, min(7, bottom))

    if right < left:
        right = left

    if bottom < top:
        bottom = top

    values = []

    for row in range(top, bottom + 1):

        for col in range(left, right + 1):

            index = row * 8 + col

            value = distances[index]

            if value > 0 and value < 8000:

                values.append(value)

    if not values:

        return None

    return statistics.median(values)


# ============================================================
# DIRECTION
# ============================================================

def get_direction(
    center_x,
    frame_width
):

    position = center_x / frame_width

    if position < 0.35:

        return "left"

    elif position > 0.65:

        return "right"

    else:

        return "ahead"


# ============================================================
# THREAT SCORE
# ============================================================

def calculate_threat(
    label,
    confidence,
    distance,
    area_ratio
):

    score = PRIORITY.get(
        label,
        0
    )

    score += confidence * 30

    if distance is not None:

        if distance < 500:

            score += 100

        elif distance < 1000:

            score += 80

        elif distance < 1500:

            score += 60

        elif distance < 2000:

            score += 40

        elif distance < 3000:

            score += 20

    score += min(
        area_ratio * 50,
        30
    )

    return score


# ============================================================
# SMART ANNOUNCEMENT LOGIC
# ============================================================

def should_announce(
    label,
    direction,
    distance
):

    global last_announced_label
    global last_announced_direction
    global last_announced_distance
    global last_announcement_time
    global last_urgent_time

    now = time.time()

    # --------------------------------------------------------
    # Nothing previously announced.
    # --------------------------------------------------------

    if last_announced_label is None:

        return True

    # --------------------------------------------------------
    # Is this a different object?
    # --------------------------------------------------------

    different_object = (
        label != last_announced_label
        or
        direction != last_announced_direction
    )

    # --------------------------------------------------------
    # URGENT CLOSE OBJECT
    # --------------------------------------------------------

    if (
        distance is not None
        and distance <= URGENT_DISTANCE_MM
    ):

        # Don't repeatedly scream the same warning.
        if now - last_urgent_time >= 2.0:

            last_urgent_time = now

            return True

        return False

    # --------------------------------------------------------
    # Normal minimum announcement gap.
    # --------------------------------------------------------

    if (
        now - last_announcement_time
        < MIN_ANNOUNCEMENT_GAP
    ):

        return False

    # --------------------------------------------------------
    # New object gets announced after cooldown.
    # --------------------------------------------------------

    if different_object:

        return True

    # --------------------------------------------------------
    # Same object:
    # only speak again if it has gotten
    # meaningfully closer.
    # --------------------------------------------------------

    if (
        distance is not None
        and last_announced_distance is not None
    ):

        distance_change = (
            last_announced_distance
            - distance
        )

        if distance_change >= DISTANCE_CHANGE_MM:

            return True

    return False


# ============================================================
# ANNOUNCE
# ============================================================

def announce(
    label,
    direction,
    distance
):

    global last_announced_label
    global last_announced_direction
    global last_announced_distance
    global last_announcement_time

    if not should_announce(
        label,
        direction,
        distance
    ):

        return

    if distance is not None:

        meters = distance / 1000.0

        message = (
            f"{label} {direction}, "
            f"{meters:.1f} meters"
        )

    else:

        message = (
            f"{label} {direction}"
        )

    print(
        "ANNOUNCE:",
        message
    )

    last_announced_label = label
    last_announced_direction = direction
    last_announced_distance = distance
    last_announcement_time = time.time()

    speak(message)


# ============================================================
# MAIN
# ============================================================

def main():

    global running

    print()
    print("==============================")
    print("       SMART WALKING STICK")
    print("       Raspberry Pi 3 B+")
    print("==============================")
    print()

    # --------------------------------------------------------
    # YOLO
    # --------------------------------------------------------

    print("Loading YOLO ONNX...")

    net = cv2.dnn.readNetFromONNX(
        MODEL_PATH
    )

    print("YOLO: READY")

    # --------------------------------------------------------
    # START THREADS
    # --------------------------------------------------------

    camera_thread = threading.Thread(
        target=camera_worker,
        daemon=True
    )

    tof_thread = threading.Thread(
        target=tof_worker,
        daemon=True
    )

    speech_thread = threading.Thread(
        target=speech_worker,
        daemon=True
    )

    camera_thread.start()
    tof_thread.start()
    speech_thread.start()

    time.sleep(2)

    print()
    print("SYSTEM READY")
    print()

    frame_counter = 0

    try:

        while True:

            # ------------------------------------------------
            # CAMERA
            # ------------------------------------------------

            with frame_lock:

                if latest_frame is None:

                    continue

                frame = latest_frame.copy()

            frame_counter += 1

            if (
                frame_counter
                % DETECTION_INTERVAL
                != 0
            ):

                continue

            frame_height, frame_width = (
                frame.shape[:2]
            )

            frame_area = (
                frame_width
                * frame_height
            )

            # ------------------------------------------------
            # TOF
            # ------------------------------------------------

            with distance_lock:

                if latest_distances is not None:

                    distances = (
                        latest_distances.copy()
                    )

                else:

                    distances = None

            # ------------------------------------------------
            # YOLO INPUT
            # ------------------------------------------------

            blob = cv2.dnn.blobFromImage(
                frame,
                1 / 255.0,
                (
                    YOLO_SIZE,
                    YOLO_SIZE
                ),
                swapRB=True,
                crop=False
            )

            net.setInput(blob)

            outputs = net.forward()

            detections = outputs[0]

            if len(detections.shape) == 3:

                detections = detections[0]

            if detections.shape[0] == 84:

                detections = detections.T

            best_object = None

            # ------------------------------------------------
            # PROCESS DETECTIONS
            # ------------------------------------------------

            for detection in detections:

                if len(detection) < 5:

                    continue

                x = float(detection[0])
                y = float(detection[1])
                w = float(detection[2])
                h = float(detection[3])

                class_scores = (
                    detection[4:]
                )

                class_id = int(
                    class_scores.argmax()
                )

                confidence = float(
                    class_scores[class_id]
                )

                if (
                    confidence
                    < CONFIDENCE_THRESHOLD
                ):

                    continue

                if class_id >= len(
                    CLASS_NAMES
                ):

                    continue

                label = CLASS_NAMES[
                    class_id
                ]

                # Only process objects
                # that matter to the cane.
                if label not in IMPORTANT_OBJECTS:

                    continue

                # ------------------------------------------------
                # BOX COORDINATES

                # ------------------------------------------------

                scale_x = (
                    frame_width
                    / YOLO_SIZE
                )

                scale_y = (
                    frame_height
                    / YOLO_SIZE
                )

                center_x = (
                    x * scale_x
                )

                center_y = (
                    y * scale_y
                )

                box_width = (
                    w * scale_x
                )

                box_height = (
                    h * scale_y
                )

                x1 = (
                    center_x
                    - box_width / 2
                )

                y1 = (
                    center_y
                    - box_height / 2
                )

                x2 = (
                    center_x
                    + box_width / 2
                )

                y2 = (
                    center_y
                    + box_height / 2
                )

                x1 = max(
                    0,
                    min(
                        frame_width,
                        x1
                    )
                )

                x2 = max(
                    0,
                    min(
                        frame_width,
                        x2
                    )
                )

                y1 = max(
                    0,
                    min(
                        frame_height,
                        y1
                    )
                )

                y2 = max(
                    0,
                    min(
                        frame_height,
                        y2
                    )
                )

                area = (
                    (x2 - x1)
                    * (y2 - y1)
                )

                area_ratio = (
                    area / frame_area
                )

                # ------------------------------------------------
                # DISTANCE
                # ------------------------------------------------

                distance = (
                    get_object_distance(
                        distances,
                        x1,
                        y1,
                        x2,
                        y2,
                        frame_width,
                        frame_height
                    )
                )

                # ------------------------------------------------
                # DIRECTION
                # ------------------------------------------------

                direction = get_direction(
                    center_x,
                    frame_width
                )

                # ------------------------------------------------
                # THREAT SCORE
                # ------------------------------------------------

                score = (
                    calculate_threat(
                        label,
                        confidence,
                        distance,
                        area_ratio
                    )
                )

                candidate = {

                    "label": label,

                    "confidence":
                        confidence,

                    "distance":
                        distance,

                    "direction":
                        direction,

                    "score":
                        score
                }

                if (
                    best_object is None
                    or
                    score
                    > best_object["score"]
                ):

                    best_object = candidate

            # ------------------------------------------------
            # ANNOUNCE ONLY THE MOST IMPORTANT OBJECT
            # ------------------------------------------------

            if best_object is not None:

                distance = (
                    best_object[
                        "distance"
                    ]
                )


                if (
                    distance is None
                    or
                    distance
                    <= MAX_DISTANCE_MM
                ):

                    announce(

                        best_object[
                            "label"
                        ],

                        best_object[
                            "direction"
                        ],

                        distance
                    )

    except KeyboardInterrupt:

        print()
        print("Stopping...")

    except Exception as e:

        print()
        print("MAIN ERROR:")
        print(
            type(e).__name__,
            e
        )

    finally:

        running = False

        speech_queue.put(None)

        print(
            "Smart Cane stopped."
        )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    main()
