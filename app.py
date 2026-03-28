import tkinter as tk
from tkinter import filedialog
import time
import cv2
from ultralytics import YOLO
from lane import Lane
from config import *

class TrafficApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Traffic System")
        self.root.geometry("1200x750")
        self.root.configure(bg="#f0f0f0")

        # -------- MAIN FRAME --------
        main_frame = tk.Frame(root, bg="#f0f0f0")
        main_frame.pack(fill=tk.BOTH, expand=1)

        canvas = tk.Canvas(main_frame, bg="#f0f0f0", highlightthickness=0)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

        scrollbar = tk.Scrollbar(main_frame, orient=tk.VERTICAL, command=canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        canvas.configure(yscrollcommand=scrollbar.set)

        # -------- CONTAINER --------
        self.container = tk.Frame(canvas, bg="#f0f0f0")

        self.canvas_window = canvas.create_window((0, 0), window=self.container, anchor="n")

        # -------- SCROLL FIX --------
        def on_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))

        self.container.bind("<Configure>", on_configure)

        # -------- CENTER FIX --------
        def center_content(event):
            canvas_width = event.width
            frame_width = self.container.winfo_reqwidth()

            x = max((canvas_width - frame_width) // 2, 0)
            canvas.coords(self.canvas_window, x, 0)

        canvas.bind("<Configure>", center_content)

        # -------- MOUSE SCROLL --------
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # -------- MODEL --------
        self.model = YOLO("yolov8n.pt")

        # -------- HEADER --------
        header = tk.Frame(self.container, bg="#f0f0f0")
        header.pack(pady=10)

        self.timer = tk.Label(header, text="", font=("Arial",16,"bold"), bg="#f0f0f0")
        self.timer.pack()

        # -------- CONTROLS --------
        control = tk.Frame(self.container, bg="#f0f0f0")
        control.pack(pady=10)

        tk.Button(control, text="Load 4 Videos", width=15, command=self.load_videos).pack(side=tk.LEFT, padx=5)
        tk.Button(control, text="Next Lane", width=12, command=self.force_next).pack(side=tk.LEFT, padx=5)
        tk.Button(control, text="Force Green", width=12, command=self.force_green).pack(side=tk.LEFT, padx=5)
        tk.Button(control, text="Force Red", width=12, command=self.force_red).pack(side=tk.LEFT, padx=5)

        # -------- GRID --------
        grid_wrapper = tk.Frame(self.container, bg="#f0f0f0")
        grid_wrapper.pack()

        grid = tk.Frame(grid_wrapper, bg="#f0f0f0")
        grid.pack()

        self.lanes = []

        for i in range(4):
            pos = "left" if i % 2 == 0 else "right"

            frame = tk.Frame(grid, bg="white", bd=2, relief="solid")
            frame.grid(row=i//2, column=i%2, padx=20, pady=20)

            lane = Lane(frame, self.model, pos)
            self.lanes.append(lane)

        # -------- TRAFFIC STATE --------
        self.current_lane = 0
        self.state = "GREEN"
        self.last_switch = time.time()

        self.loop()

    def load_videos(self):
        paths = filedialog.askopenfilenames(filetypes=[("Video","*.mp4")])
        if len(paths) != 4:
            print("Select 4 videos")
            return

        for lane, p in zip(self.lanes, paths):
            lane.cap = cv2.VideoCapture(p)
            lane.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

        self.lanes[0].set_green()
        self.last_switch = time.time()

    def calculate_green_time(self):
        density = sum(self.lanes[self.current_lane].counts.values())
        return max(MIN_GREEN, min(MAX_GREEN, int(5 + density * 0.5)))

    def force_next(self):
        self.lanes[self.current_lane].set_red()
        self.current_lane = (self.current_lane + 1) % 4
        self.lanes[self.current_lane].set_green()
        self.state = "GREEN"
        self.last_switch = time.time()

    def force_green(self):
        for lane in self.lanes:
            lane.set_red()
        self.lanes[self.current_lane].set_green()
        self.state = "GREEN"
        self.last_switch = time.time()

    def force_red(self):
        for lane in self.lanes:
            lane.set_red()
        self.state = "RED"

    def loop(self):
        elapsed = time.time() - self.last_switch

        if self.state == "GREEN":
            dyn = self.calculate_green_time()
            rem = int(dyn - elapsed)

            self.timer.config(
                text=f"Lane {self.current_lane+1} GREEN {rem}s | Density {sum(self.lanes[self.current_lane].counts.values())}"
            )

            if elapsed > dyn:
                self.state = "YELLOW"
                self.last_switch = time.time()
                self.lanes[self.current_lane].set_yellow()

        elif self.state == "YELLOW":
            rem = int(YELLOW_TIME - elapsed)
            self.timer.config(text=f"Lane {self.current_lane+1} YELLOW {rem}s")

            if elapsed > YELLOW_TIME:
                self.lanes[self.current_lane].set_red()
                self.current_lane = (self.current_lane + 1) % 4
                self.lanes[self.current_lane].set_green()
                self.state = "GREEN"
                self.last_switch = time.time()

        for lane in self.lanes:
            lane.update_ui()

        self.root.after(30, self.loop)


if __name__ == "__main__":
    root = tk.Tk()
    app = TrafficApp(root)
    root.mainloop()