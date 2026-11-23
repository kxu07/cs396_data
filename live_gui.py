"""
Live glove pose GUI: reads the sensor stream from the Arduino and shows the
predicted pose in real time.

Uses the same serial packet format as sensor_logger.py and the model saved by
train_glove.py. Live inputs are processed like the training data:
  1. Calibrate: for ~10 s, open and close your hand fully a few times.
     Each finger is then scaled with the 2nd/98th percentile of that data.
  2. Every 100 ms, average the last 100 ms of samples per finger.
  3. Standardize with the training mean/std and run the MLP.
  4. Average the probabilities over the last few predictions to smooth.

Usage:
    python live_gui.py --port /dev/cu.usbmodem101
    python live_gui.py --port /dev/cu.usbmodem101 --model-dir results
    python live_gui.py --replay jay1.csv        # test without the glove

Needs predict_pose.py and sensor_logger.py in the same folder.
Keys: C = calibrate, Q / Esc = quit.
"""

import argparse
import threading
import time
from collections import deque

import numpy as np

from predict_pose import load_model, predict

# Order the Arduino sends the values in (same as the CSV columns).
STREAM_ORDER = ["index", "middle", "ring", "pinky", "thumb"]
POSE_COLORS = ["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854", "#ffd92f"]


# ---------------------------------------------------------------------------
# Thread-safe ring buffer of (time, 5 values)
# ---------------------------------------------------------------------------
class RingBuffer:
    def __init__(self, capacity, n_values):
        self.t = np.zeros(capacity)
        self.x = np.zeros((capacity, n_values))
        self.capacity = capacity
        self.count = 0
        self.lock = threading.Lock()

    def append(self, t, values):
        with self.lock:
            i = self.count % self.capacity
            self.t[i] = t
            self.x[i] = values
            self.count += 1

    def since(self, t_from):
        """All buffered samples with time >= t_from, oldest first."""
        with self.lock:
            n = min(self.count, self.capacity)
            idx = np.arange(self.count - n, self.count) % self.capacity
            t, x = self.t[idx], self.x[idx]
        keep = t >= t_from
        return t[keep], x[keep]

    def last_time(self):
        with self.lock:
            if self.count == 0:
                return None
            return self.t[(self.count - 1) % self.capacity]


# ---------------------------------------------------------------------------
# Data sources (background threads that fill the buffer)
# ---------------------------------------------------------------------------
class SerialSource(threading.Thread):
    """Reads packets from the Arduino exactly like sensor_logger.py."""

    def __init__(self, port, baud, buf):
        super().__init__(daemon=True)
        self.port, self.baud, self.buf = port, baud, buf
        self.status = "Connecting..."
        self.error = None
        self.stop_event = threading.Event()

    def run(self):
        import serial
        from sensor_logger import read_packet
        try:
            with serial.Serial(self.port, self.baud, timeout=1) as ser:
                time.sleep(2)              # Arduino resets when the port opens
                ser.reset_input_buffer()
                self.status = f"Streaming from {self.port}"
                t0 = time.perf_counter()
                while not self.stop_event.is_set():
                    values = read_packet(ser)
                    if values is not None:
                        self.buf.append(time.perf_counter() - t0, values)
        except Exception as e:  # port missing, unplugged, etc.
            self.error = str(e)
            self.status = f"Serial error: {e}"


class ReplaySource(threading.Thread):
    """Plays a recorded CSV back in real time (loops forever), for testing."""

    def __init__(self, csv_path, buf, speed=1.0):
        super().__init__(daemon=True)
        import pandas as pd
        df = pd.read_csv(csv_path)
        self.t = df["time_s"].to_numpy(float)
        self.x = df[STREAM_ORDER].to_numpy(float)
        self.buf, self.speed = buf, speed
        self.status = f"Replaying {csv_path}"
        self.error = None
        self.stop_event = threading.Event()

    def run(self):
        duration = self.t[-1] + 0.001
        t0 = time.perf_counter()
        i, loop = 0, 0
        while not self.stop_event.is_set():
            now = (time.perf_counter() - t0) * self.speed
            while i < len(self.t) and self.t[i] + loop * duration <= now:
                self.buf.append(self.t[i] + loop * duration, self.x[i])
                i += 1
            if i == len(self.t):
                i, loop = 0, loop + 1
            time.sleep(0.005)


# ---------------------------------------------------------------------------
# Calibration + prediction (no GUI code, so it can be tested on its own)
# ---------------------------------------------------------------------------
class LivePredictor:
    def __init__(self, model_dir, smooth=3):
        self.model, self.cfg = load_model(model_dir)
        self.pose_names = self.cfg["pose_names"]
        self.window_s = self.cfg.get("window_ms", 100) / 1000.0
        # Reorder the stream's columns into the order the model expects.
        self.model_idx = [STREAM_ORDER.index(s) for s in self.cfg["sensors"]]
        self.lo = self.hi = None
        self.history = deque(maxlen=max(1, smooth))

    @property
    def calibrated(self):
        return self.lo is not None

    def calibrate(self, x):
        """x: raw samples (n, 5) in stream order, recorded while opening/closing the hand."""
        x = x[:, self.model_idx]
        self.lo = np.percentile(x, 2, axis=0)
        self.hi = np.percentile(x, 98, axis=0)
        self.history.clear()

    def predict(self, x_window):
        """x_window: raw samples (n, 5) from the last window. Returns smoothed probs."""
        mean = x_window[:, self.model_idx].mean(axis=0)
        span = np.where(self.hi - self.lo == 0, 1, self.hi - self.lo)
        cal = np.clip((mean - self.lo) / span, 0, 1)
        self.history.append(predict(self.model, self.cfg, cal[None, :])[0])
        return np.mean(self.history, axis=0)


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------
class App:
    TICK_MS = 100
    PLOT_S = 5.0

    def __init__(self, source, buf, predictor, calib_s=10.0, min_conf=0.5):
        import tkinter as tk
        from tkinter import ttk
        import matplotlib
        matplotlib.use("TkAgg")
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        self.source, self.buf, self.pred = source, buf, predictor
        self.calib_s, self.min_conf = calib_s, min_conf
        self.calib_start = None
        self.rate_count, self.rate_time, self.rate = 0, time.perf_counter(), 0.0

        self.root = tk.Tk()
        self.root.title("Glove pose — live")
        self.root.geometry("900x700")
        self.root.protocol("WM_DELETE_WINDOW", self.quit)
        self.root.bind("<KeyPress-c>", lambda e: self.start_calibration())
        self.root.bind("<KeyPress-q>", lambda e: self.quit())
        self.root.bind("<Escape>", lambda e: self.quit())

        # Big pose label
        self.pose_var = tk.StringVar(value="Waiting for data...")
        self.pose_label = tk.Label(self.root, textvariable=self.pose_var,
                                   font=("Helvetica", 40, "bold"), bg="#eeeeee",
                                   height=2)
        self.pose_label.pack(fill="x", padx=10, pady=(10, 0))
        self.conf_var = tk.StringVar(value="")
        tk.Label(self.root, textvariable=self.conf_var,
                 font=("Helvetica", 14)).pack(pady=(4, 6))

        # Probability bars
        bars = tk.Frame(self.root)
        bars.pack(fill="x", padx=20)
        self.bar_vars, self.pct_vars = [], []
        for i, name in enumerate(self.pred.pose_names):
            tk.Label(bars, text=name, width=14, anchor="w",
                     font=("Helvetica", 12)).grid(row=i, column=0, sticky="w")
            v = tk.DoubleVar(value=0)
            ttk.Progressbar(bars, variable=v, maximum=100,
                            length=500).grid(row=i, column=1, sticky="ew", pady=2)
            p = tk.StringVar(value="")
            tk.Label(bars, textvariable=p, width=6,
                     font=("Helvetica", 12)).grid(row=i, column=2)
            self.bar_vars.append(v)
            self.pct_vars.append(p)
        bars.columnconfigure(1, weight=1)

        # Buttons
        btns = tk.Frame(self.root)
        btns.pack(pady=8)
        tk.Button(btns, text="Calibrate (C)", width=16,
                  command=self.start_calibration).pack(side="left", padx=5)
        tk.Button(btns, text="Quit (Q)", width=10,
                  command=self.quit).pack(side="left", padx=5)

        # Live sensor plot
        fig = Figure(figsize=(9, 2.6), dpi=100)
        self.ax = fig.add_subplot(111)
        self.lines = [self.ax.plot([], [], linewidth=1, label=s)[0] for s in STREAM_ORDER]
        self.ax.set_ylim(0, 1023)
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Reading")
        self.ax.grid(alpha=0.3)
        self.ax.legend(loc="upper left", ncol=5, fontsize=8)
        fig.tight_layout()
        self.canvas = FigureCanvasTkAgg(fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=10)

        # Status bar
        self.status_var = tk.StringVar(value="")
        tk.Label(self.root, textvariable=self.status_var, anchor="w",
                 relief="sunken").pack(fill="x", side="bottom")

        self.root.after(self.TICK_MS, self.tick)

    # -- actions --------------------------------------------------------------
    def start_calibration(self):
        t_last = self.buf.last_time()
        if t_last is None:
            return
        self.calib_start = t_last

    def quit(self):
        self.source.stop_event.set()
        self.root.destroy()

    def run(self):
        self.root.mainloop()

    # -- update loop ----------------------------------------------------------
    def set_pose(self, text, color="#eeeeee"):
        self.pose_var.set(text)
        self.pose_label.config(bg=color)

    def tick(self):
        try:
            self.update()
        finally:
            self.root.after(self.TICK_MS, self.tick)

    def update(self):
        # Samples per second
        count = self.buf.count
        now = time.perf_counter()
        if now - self.rate_time >= 1.0:
            self.rate = (count - self.rate_count) / (now - self.rate_time)
            self.rate_count, self.rate_time = count, now
        cal_txt = "calibrated" if self.pred.calibrated else "not calibrated"
        self.status_var.set(f"  {self.source.status}   |   {self.rate:.0f} samples/s   |   {cal_txt}")

        t_last = self.buf.last_time()
        if t_last is None:
            return

        self.update_plot(t_last)

        # Calibration in progress
        if self.calib_start is not None:
            remaining = self.calib_s - (t_last - self.calib_start)
            if remaining > 0:
                self.set_pose(f"Open & close your hand… {remaining:.0f}s", "#fff3b0")
                self.conf_var.set("Calibrating: fully open and fully close a few times")
                return
            _, x = self.buf.since(self.calib_start)
            self.pred.calibrate(x)
            self.calib_start = None

        if not self.pred.calibrated:
            self.set_pose("Press Calibrate to start")
            self.conf_var.set("")
            return

        # Predict from the last window
        _, x = self.buf.since(t_last - self.pred.window_s)
        if len(x) == 0:
            return
        probs = self.pred.predict(x)
        best = int(np.argmax(probs))
        conf = probs[best]
        for i, p in enumerate(probs):
            self.bar_vars[i].set(p * 100)
            self.pct_vars[i].set(f"{p * 100:.0f}%")
        if conf < self.min_conf:
            self.set_pose("Uncertain")
        else:
            self.set_pose(self.pred.pose_names[best], POSE_COLORS[best % len(POSE_COLORS)])
        self.conf_var.set(f"Confidence {conf * 100:.0f}%")

    def update_plot(self, t_last):
        t, x = self.buf.since(t_last - self.PLOT_S)
        step = max(1, len(t) // 1500)  # thin out points to keep drawing fast
        for j, line in enumerate(self.lines):
            line.set_data(t[::step], x[::step, j])
        self.ax.set_xlim(t_last - self.PLOT_S, t_last)
        self.canvas.draw_idle()


def main():
    ap = argparse.ArgumentParser(description="Live glove pose GUI.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--port", help="Serial port, e.g. COM3 or /dev/cu.usbmodem101")
    src.add_argument("--replay", help="Play a recorded CSV instead of the glove (for testing)")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--model-dir", default="results",
                    help="Folder with glove_mlp.pt and glove_mlp_config.json")
    ap.add_argument("--calib-s", type=float, default=10, help="Calibration length in seconds")
    ap.add_argument("--smooth", type=int, default=3,
                    help="Average over this many predictions (1 = no smoothing)")
    ap.add_argument("--min-conf", type=float, default=0.5,
                    help="Show 'Uncertain' below this confidence (0..1)")
    args = ap.parse_args()

    buf = RingBuffer(capacity=60000, n_values=len(STREAM_ORDER))  # ~50 s at 1.2 kHz
    predictor = LivePredictor(args.model_dir, smooth=args.smooth)
    source = (SerialSource(args.port, args.baud, buf) if args.port
              else ReplaySource(args.replay, buf))
    source.start()
    App(source, buf, predictor, calib_s=args.calib_s, min_conf=args.min_conf).run()


if __name__ == "__main__":
    main()