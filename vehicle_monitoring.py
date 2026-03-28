import cv2
import csv
import numpy as np
from ultralytics import YOLO
from sort import Sort, iou
import tkinter as tk
from tkinter import filedialog

# =========================
# CONFIG
# =========================
MODEL_PATH = "yolov8n.pt"
CONF_THRESH = 0.3
LINE_Y = 420
PIXELS_PER_METER = 8   # MUST CALIBRATE FOR YOUR CAMERA

VEHICLE_IDS = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

# =========================
# VIDEO PICKER
# =========================
tk.Tk().withdraw()
VIDEO_PATH = filedialog.askopenfilename(
    title="Select Video File",
    filetypes=[("Video Files", "*.mp4 *.avi *.mov")]
)

if not VIDEO_PATH:
    print("❌ No video selected")
    exit()

# =========================
# INITIALIZATION
# =========================
model = YOLO(MODEL_PATH)
tracker = Sort(iou_threshold=0.3)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("❌ Cannot open video")
    exit()

FPS = cap.get(cv2.CAP_PROP_FPS)

track_history = {}      # track_id -> [(cx, cy)]
track_classes = {}      # track_id -> class
counted_ids = set()

paused = False

# =========================
# CSV
# =========================
csv_file = open("vehicle_logs.csv", "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Track ID", "Vehicle Type", "Speed (km/h)"])

# =========================
# MAIN LOOP
# =========================
while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (1280, 720))
        results = model(frame, verbose=False)[0]

        detections = []
        det_classes = []

        if results.boxes is not None:
            for box in results.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                if cls in VEHICLE_IDS and conf >= CONF_THRESH:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detections.append([x1, y1, x2, y2])
                    det_classes.append(cls)

        detections_np = np.array(detections) if detections else np.empty((0, 4))
        tracks = tracker.update(detections_np)

        # Draw counting line
        cv2.line(frame, (0, LINE_Y), (1280, LINE_Y), (0, 0, 255), 2)

        for trk in tracks:
            x1, y1, x2, y2, track_id = map(int, trk)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # =========================
            # CLASS ASSOCIATION (IOU-BASED)
            # =========================
            if track_id not in track_classes and len(detections) > 0:
                best_iou = 0
                best_cls = None
                for det, cls in zip(detections, det_classes):
                    score = iou([x1, y1, x2, y2], det)
                    if score > best_iou:
                        best_iou = score
                        best_cls = cls
                if best_cls is not None:
                    track_classes[track_id] = VEHICLE_IDS[best_cls]

            vehicle_type = track_classes.get(track_id, "vehicle")

            # =========================
            # TRACK HISTORY
            # =========================
            if track_id not in track_history:
                track_history[track_id] = []

            track_history[track_id].append((cx, cy))

            # =========================
            # SPEED ESTIMATION
            # =========================
            speed = 0
            if len(track_history[track_id]) >= 2:
                x_prev, y_prev = track_history[track_id][-2]
                dist_px = np.hypot(cx - x_prev, cy - y_prev)
                speed = (dist_px / PIXELS_PER_METER) * FPS * 3.6

            # =========================
            # LINE CROSS COUNT
            # =========================
            if len(track_history[track_id]) >= 2:
                prev_cy = track_history[track_id][-2][1]
                if prev_cy < LINE_Y and cy >= LINE_Y and track_id not in counted_ids:
                    counted_ids.add(track_id)
                    csv_writer.writerow([track_id, vehicle_type, int(speed)])

            # =========================
            # DRAW
            # =========================
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{vehicle_type} | ID:{track_id} | {int(speed)} km/h",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

        # =========================
        # UI PANEL
        # =========================
        cv2.rectangle(frame, (0, 0), (380, 90), (0, 0, 0), -1)
        cv2.putText(frame, f"Total Vehicles: {len(counted_ids)}",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.putText(frame, "SPACE: Pause | ESC: Exit",
                    (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("Vehicle Traffic Monitoring", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key == 32:
        paused = not paused

# =========================
# CLEANUP
# =========================
cap.release()
csv_file.close()
cv2.destroyAllWindows()
