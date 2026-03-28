import tkinter as tk
from tkinter import filedialog
import time
import cv2
from ultralytics import YOLO
from lane import Lane
from config import *
from datetime import datetime
from dashboard import Dashboard


class TrafficApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Smart Traffic Control System")
        self.root.geometry("1980x1024")

        # ================= HEADER =================
        self.header = tk.Frame(root, bg="#0f172a", height=80)
        self.header.pack(fill="x")

        tk.Label(self.header, text="🚦 AI Smart Traffic Control System",
                 fg="white", bg="#0f172a",
                 font=("Segoe UI", 20, "bold")).place(x=30, y=20)

        self.timer = tk.Label(self.header,
                              fg="#22c55e", bg="#0f172a",
                              font=("Segoe UI", 13, "bold"))
        self.timer.place(x=30, y=55)

        tk.Button(self.header, text="Live",
                  bg="#1e293b", fg="white",
                  command=lambda: self.show_frame(self.live_frame)).place(relx=0.4, rely=0.3)

        tk.Button(self.header, text="Dashboard",
                  bg="#1e293b", fg="white",
                  command=lambda: self.show_frame(self.dashboard_frame)).place(relx=0.5, rely=0.3)

        self.clock = tk.Label(self.header,
                              fg="#cbd5f5", bg="#0f172a")
        self.clock.place(relx=0.85, rely=0.2)

        self.global_alert = tk.Label(self.header,
                                    fg="red", bg="#0f172a")
        self.global_alert.place(relx=0.7, rely=0.5)

        # ================= MAIN =================
        self.container = tk.Frame(root)
        self.container.pack(fill="both", expand=True)

        # CONTROLS
        control = tk.Frame(self.container)
        control.pack(pady=10)

        tk.Button(control, text="Load 4 Videos", command=self.load_videos).pack(side=tk.LEFT, padx=5)
        tk.Button(control, text="Next Lane", command=self.force_next).pack(side=tk.LEFT, padx=5)
        tk.Button(control, text="Force Green", command=self.force_green).pack(side=tk.LEFT, padx=5)
        tk.Button(control, text="Force Red", command=self.force_red).pack(side=tk.LEFT, padx=5)

        # ================= FRAMES =================
        self.main_area = tk.Frame(self.container)
        self.main_area.pack(fill="both", expand=True)

        self.live_frame = tk.Frame(self.main_area)
        self.dashboard_frame = tk.Frame(self.main_area, bg="#f8fafc")

        for f in (self.live_frame, self.dashboard_frame):
            f.grid(row=0, column=0, sticky="nsew")

        # ================= LIVE FIX =================
        live_wrapper = tk.Frame(self.live_frame)
        live_wrapper.pack(fill="both", expand=True)

        # USE GRID FOR TRUE CENTER
        live_wrapper.grid_rowconfigure(0, weight=1)
        live_wrapper.grid_columnconfigure(0, weight=1)

        center = tk.Frame(live_wrapper)
        center.grid(row=0, column=0)

        grid = tk.Frame(center)
        grid.pack(expand=True)

        self.model = YOLO("yolov8n.pt")

        self.lanes = []
        for i in range(4):
            frame = tk.Frame(
                grid, bg="white", bd=2, relief="solid",
                width=640, height=480  # Larger video box
            )
            frame.grid(row=i//2, column=i%2, padx=40, pady=40, sticky="nsew")
            frame.grid_propagate(False)  # Prevent shrinking

            pos = "left" if i % 2 == 0 else "right"
            self.lanes.append(Lane(frame, self.model, pos))

        # Make grid cells expand equally
        grid.grid_rowconfigure(0, weight=1)
        grid.grid_rowconfigure(1, weight=1)
        grid.grid_columnconfigure(0, weight=1)
        grid.grid_columnconfigure(1, weight=1)

        # ================= DASHBOARD =================
        self.dashboard = Dashboard(self.dashboard_frame, self.lanes)

        # STATE
        self.current_lane = 0
        self.state = "GREEN"
        self.last_switch = time.time()

        self.show_frame(self.live_frame)
        self.update_clock()
        self.loop()

    def show_frame(self, frame):
        frame.tkraise()

    def update_clock(self):
        self.clock.config(text=datetime.now().strftime("%H:%M:%S"))
        self.root.after(1000, self.update_clock)

    def load_videos(self):
        paths = filedialog.askopenfilenames(filetypes=[("Video", "*.mp4")])
        if len(paths) != 4:
            return

        for lane, p in zip(self.lanes, paths):
            lane.cap = cv2.VideoCapture(p)

        self.lanes[0].set_green()
        self.last_switch = time.time()

    def calculate_green_time(self):
        density = sum(self.lanes[self.current_lane].counts.values())
        return max(MIN_GREEN, min(MAX_GREEN, int(5 + density * 0.5)))

    def force_next(self):
        self.lanes[self.current_lane].set_red()
        self.current_lane = (self.current_lane + 1) % 4
        self.lanes[self.current_lane].set_green()
        self.last_switch = time.time()

    def force_green(self):
        for l in self.lanes:
            l.set_red()
        self.lanes[self.current_lane].set_green()

    def force_red(self):
        for l in self.lanes:
            l.set_red()

    def loop(self):
        elapsed = time.time() - self.last_switch

        if self.state == "GREEN":
            dyn = self.calculate_green_time()
            rem = int(dyn - elapsed)

            self.timer.config(text=f"Lane {self.current_lane+1} GREEN {rem}s")

            if elapsed > dyn:
                self.state = "YELLOW"
                self.last_switch = time.time()
                self.lanes[self.current_lane].set_yellow()

        elif self.state == "YELLOW":
            if elapsed > YELLOW_TIME:
                self.lanes[self.current_lane].set_red()
                self.current_lane = (self.current_lane + 1) % 4
                self.lanes[self.current_lane].set_green()
                self.state = "GREEN"
                self.last_switch = time.time()

        for lane in self.lanes:
            lane.update_ui()

        self.dashboard.update()

        if any(l.accident for l in self.lanes):
            self.global_alert.config(text="⚠ ACCIDENT DETECTED")
        else:
            self.global_alert.config(text="")

        self.root.after(10, self.loop)


if __name__ == "__main__":
    root = tk.Tk()
    app = TrafficApp(root)
    root.mainloop()