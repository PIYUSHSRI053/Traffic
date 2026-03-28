import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk
from ultralytics import YOLO
from sort import Sort
import csv
import os

# ================= CONFIG =================
CONF_THRESH = 0.3
LINE_Y = 260
TRUCK_AREA_THRESHOLD = 45000

# COCO vehicle classes (EXPLICIT — VERY IMPORTANT)
ALLOWED_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}
# =========================================

class VehicleApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Vehicle Traffic Monitoring System")
        self.root.geometry("1000x700")

        self.model = YOLO("yolov8n.pt")
        self.tracker = Sort()

        self.cap = None
        self.running = False

        self.counted_ids = set()
        self.counts = {
            "car": 0,
            "motorcycle": 0,
            "bus": 0,
            "truck": 0
        }

        os.makedirs("logs", exist_ok=True)
        self.csv = open("logs/vehicle_logs.csv", "w", newline="")
        self.writer = csv.writer(self.csv)
        self.writer.writerow(["ID", "Vehicle_Type"])

        # ---------------- UI ----------------
        tk.Label(root, text="Vehicle Traffic Monitoring System",
                 font=("Arial", 16, "bold")).pack(pady=5)

        self.video_label = tk.Label(root)
        self.video_label.pack()

        self.info = tk.Label(root, text="", font=("Arial", 12))
        self.info.pack(pady=5)

        btns = tk.Frame(root)
        btns.pack(pady=10)

        tk.Button(btns, text="Choose File", command=self.load_video).pack(side=tk.LEFT, padx=5)
        tk.Button(btns, text="Start", bg="green", fg="white",
                  command=self.start).pack(side=tk.LEFT, padx=5)
        tk.Button(btns, text="Stop", bg="red", fg="white",
                  command=self.stop).pack(side=tk.LEFT, padx=5)

    # ---------------- FUNCTIONS ----------------
    def load_video(self):
        path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4")])
        if path:
            self.cap = cv2.VideoCapture(path)

    def start(self):
        if self.cap:
            self.running = True
            self.update()

    def stop(self):
        self.running = False

    def update(self):
        if not self.running:
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.resize(frame, (800, 450))
        results = self.model(frame, verbose=False)[0]

        detections = []
        class_map = {}

        # ---------- DETECTION ----------
        if results.boxes:
            for box in results.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                if conf < CONF_THRESH or cls_id not in ALLOWED_CLASSES:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = ALLOWED_CLASSES[cls_id]

                area = (x2 - x1) * (y2 - y1)

                # SAFE truck → car correction (DOES NOT BREAK BIKE)
                if label == "truck" and 15000 < area < TRUCK_AREA_THRESHOLD:
                    label = "car"

                detections.append([x1, y1, x2, y2, conf])
                class_map[(x1, y1, x2, y2)] = label

        tracks = self.tracker.update(np.array(detections)) if detections else []

        cv2.line(frame, (0, LINE_Y), (800, LINE_Y), (0, 0, 255), 2)

        # ---------- TRACKING + COUNTING ----------
        for trk in tracks:
            x1, y1, x2, y2, conf, tid = trk
            x1, y1, x2, y2, tid = map(int, [x1, y1, x2, y2, tid])
            cy = (y1 + y2) // 2

            vtype = class_map.get((x1, y1, x2, y2))
            if vtype is None:
                continue  # DO NOT DEFAULT TO CAR

            if cy > LINE_Y and tid not in self.counted_ids:
                self.counted_ids.add(tid)
                self.counts[vtype] += 1
                self.writer.writerow([tid, vtype])

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{vtype} ID:{tid}",
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 2)

        total = sum(self.counts.values())
        self.info.config(
            text=f"Total: {total} | Cars: {self.counts['car']} | "
                 f"Bikes: {self.counts['motorcycle']} | "
                 f"Buses: {self.counts['bus']} | "
                 f"Trucks: {self.counts['truck']}"
        )

        img = ImageTk.PhotoImage(
            Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        )
        self.video_label.imgtk = img
        self.video_label.configure(image=img)

        self.root.after(30, self.update)

# ---------------- RUN ----------------
root = tk.Tk()
VehicleApp(root)
root.mainloop()
