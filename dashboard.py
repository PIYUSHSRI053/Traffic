import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk

class Dashboard:
    def __init__(self, parent, lanes):
        self.lanes = lanes
        self.history = []

        self.wrapper = tk.Frame(parent, bg="#f8fafc")
        self.wrapper.pack(fill="both", expand=True)

        # GRID LAYOUT
        self.grid = tk.Frame(self.wrapper, bg="#f8fafc")
        self.grid.pack(expand=True)

        self.fig, self.axs = plt.subplots(2, 2, figsize=(12,7))
        self.fig.patch.set_facecolor("white")

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.grid)
        self.canvas.get_tk_widget().pack(expand=True)

    def style_ax(self, ax):
        ax.set_facecolor("white")
        ax.grid(True, linestyle="--", alpha=0.3)

    def update(self):
        values = [sum(l.counts.values()) for l in self.lanes]
        self.history.append(values)

        if len(self.history) > 30:
            self.history.pop(0)

        for row in self.axs:
            for ax in row:
                ax.clear()
                self.style_ax(ax)

        # 🔹 BIGGER LINE GRAPH
        for i in range(4):
            lane_data = [h[i] for h in self.history]
            self.axs[0][0].plot(lane_data, linewidth=2, label=f"L{i+1}")
        self.axs[0][0].set_title("Traffic Trend", fontsize=12)
        self.axs[0][0].legend()

        # 🔹 BAR
        self.axs[0][1].bar(["L1","L2","L3","L4"], values)
        self.axs[0][1].set_title("Lane Load", fontsize=12)

        # 🔹 VEHICLE TYPES
        types = ["car","motorcycle","bus","truck"]
        totals = [sum(l.counts[t] for l in self.lanes) for t in types]
        self.axs[1][0].bar(types, totals)
        self.axs[1][0].set_title("Vehicle Types", fontsize=12)

        # 🔥 PIE CHART (ACCIDENT)
        accidents = [1 if l.accident else 0 for l in self.lanes]
        labels = ["L1","L2","L3","L4"]

        if sum(accidents) == 0:
            accidents = [1,0,0,0]  # avoid crash

        self.axs[1][1].pie(accidents, labels=labels, autopct='%1.0f%%')
        self.axs[1][1].set_title("Accident Distribution")

        self.fig.tight_layout()
        self.canvas.draw()