import cv2
import tkinter as tk
from sort import Sort
from config import *
from utils import convert_frame

class Lane:
    def __init__(self, parent, model, position):
        self.model = model
        self.tracker = Sort()
        self.cap = None
        self.running = False
        self.position = position

        self.counts = {"car":0,"motorcycle":0,"bus":0,"truck":0}

        self.prev_positions = {}
        self.stop_frames = {}

        self.accident = False
        self.alert_on = False

        self.last_frame = None

        # UI
        container = tk.Frame(parent, bg="white")
        container.pack()

        if position == "left":
            self.canvas = tk.Canvas(container, width=60, height=150, bg="black")
            self.canvas.grid(row=0, column=0, padx=5)

        self.video_label = tk.Label(container, bg="black")
        self.video_label.grid(row=0, column=1)

        if position == "right":
            self.canvas = tk.Canvas(container, width=60, height=150, bg="black")
            self.canvas.grid(row=0, column=2, padx=5)

        self.red = self.canvas.create_oval(10,10,50,50,fill="red")
        self.yellow = self.canvas.create_oval(10,55,50,95,fill="gray")
        self.green = self.canvas.create_oval(10,100,50,140,fill="gray")

        self.signal = tk.Label(parent, text="RED", fg="red",
                               font=("Arial",12,"bold"), bg="white")
        self.signal.pack()

        self.info = tk.Label(parent, text="", font=("Arial",10), bg="white")
        self.info.pack()

        self.alert_label = tk.Label(parent, text="", font=("Arial",12,"bold"), bg="white")
        self.alert_label.pack()

        self.stop_btn = tk.Button(parent, text="Stop Alert",
                                 command=self.stop_alert, bg="red", fg="white")
        self.stop_btn.pack(pady=5)

    def set_green(self):
        self.running = True
        self.signal.config(text="GREEN", fg="green")
        self.canvas.itemconfig(self.red, fill="gray")
        self.canvas.itemconfig(self.yellow, fill="gray")
        self.canvas.itemconfig(self.green, fill="green")

    def set_red(self):
        self.running = False
        self.signal.config(text="RED", fg="red")
        self.canvas.itemconfig(self.red, fill="red")
        self.canvas.itemconfig(self.yellow, fill="gray")
        self.canvas.itemconfig(self.green, fill="gray")

    def set_yellow(self):
        self.running = False
        self.signal.config(text="YELLOW", fg="orange")
        self.canvas.itemconfig(self.red, fill="gray")
        self.canvas.itemconfig(self.yellow, fill="yellow")
        self.canvas.itemconfig(self.green, fill="gray")

    def blink_alert(self):
        if not self.alert_on:
            return

        if self.alert_label.cget("text") == "":
            self.alert_label.config(text="⚠ ACCIDENT ALERT", fg="red")
        else:
            self.alert_label.config(text="")

        self.alert_label.after(500, self.blink_alert)

    def stop_alert(self):
        self.alert_on = False
        self.accident = False
        self.alert_label.config(text="")

    def process(self):

        if not self.cap:
            return None

        if not self.running:
            return self.last_frame

        ret, frame = self.cap.read()

        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
            if not ret:
                return None

        frame = cv2.resize(frame, (400,250))
        results = self.model(frame, verbose=False)[0]

        self.counts = {"car":0,"motorcycle":0,"bus":0,"truck":0}
        current_positions = {}

        if results.boxes:
            for i, box in enumerate(results.boxes):
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                if conf < CONF_THRESH or cls_id not in ALLOWED_CLASSES:
                    continue

                x1,y1,x2,y2 = map(int, box.xyxy[0])
                label = ALLOWED_CLASSES[cls_id]

                self.counts[label] += 1

                cx = (x1 + x2)//2
                cy = (y1 + y2)//2
                current_positions[i] = (cx, cy)

                if i in self.prev_positions:
                    px, py = self.prev_positions[i]
                    dist = ((cx-px)**2 + (cy-py)**2)**0.5

                    if dist < 3:
                        self.stop_frames[i] = self.stop_frames.get(i,0) + 1
                    else:
                        self.stop_frames[i] = 0

                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.putText(frame,label,(x1,y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)

        self.prev_positions = current_positions

        self.accident = any(v > 15 for v in self.stop_frames.values())

        if self.accident and not self.alert_on:
            self.alert_on = True
            self.blink_alert()
        elif not self.accident:
            self.alert_on = False
            self.alert_label.config(text="")

        self.last_frame = frame
        return frame

    def update_ui(self):
        frame = self.process()
        if frame is None:
            return

        img = convert_frame(frame)
        self.video_label.imgtk = img
        self.video_label.configure(image=img)

        total = sum(self.counts.values())
        self.info.config(
            text=f"Total:{total}\nCars:{self.counts['car']} Bikes:{self.counts['motorcycle']}\nBuses:{self.counts['bus']} Trucks:{self.counts['truck']}"
        )