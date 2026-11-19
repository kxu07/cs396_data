
"""
Plot every sensor column from a sensor_logger.py CSV on one graph,
with the hand pose labelled for each time range.
 
Pose ranges are read from the same ranges.txt used by combine_poses.py,
looked up by the CSV's file name (e.g. [jay1.csv]).
 
Usage:
    python plot_sensors.py run1.csv
    python plot_sensors.py run1.csv --ranges ranges.txt --save run1.png
"""
 
import argparse
import re
from pathlib import Path
 
import matplotlib.pyplot as plt
import pandas as pd
 
LINE_RE = re.compile(r"^\s*(\d+)\s+(.+?)\s*:\s*([\d.]+)\s*-\s*([\d.]+)\s*$")
 
 
def load_pose_ranges(ranges_path, csv_name):
    """Return [(pose_id, pose_name, start, end), ...] for csv_name."""
    current, found = None, []
    for line in Path(ranges_path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1].strip()
            continue
        m = LINE_RE.match(line)
        if m and current == csv_name:
            pid, name, start, end = m.groups()
            found.append((int(pid), name, float(start), float(end)))
    return found
 
 
def plot_sensors(csv_path, ranges_path="ranges.txt", save_path=None):
    """Plot all sensor columns against time, with pose ranges labelled."""
    df = pd.read_csv(csv_path)
    sensors = [col for col in df.columns if col != "time_s"]
 
    fig, ax = plt.subplots(figsize=(10, 5))
    for col in sensors:
        ax.plot(df["time_s"], df[col], label=col, linewidth=1)
 
    # Shade each pose's time range and put a floating label on it.
    poses = []
    if Path(ranges_path).exists():
        poses = load_pose_ranges(ranges_path, Path(csv_path).name)
    if not poses:
        print(f"No pose ranges for {Path(csv_path).name} in {ranges_path}; "
              "plotting without labels.")
    for pid, name, start, end in poses:
        ax.axvspan(start, end, color="gray", alpha=0.12, linewidth=0)
        ax.text((start + end) / 2, 0.97, f"{pid} {name}",
                transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="gray", alpha=0.85))
 
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Reading (0–1023)")
    ax.set_title(f"Sensor readings — {csv_path}")
    ax.set_ylim(0, 1023)
    ax.grid(alpha=0.3)
    # Legend below the plot so it doesn't cover the pose labels.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12),
              ncol=len(sensors), frameon=False)
    fig.tight_layout()
 
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved plot to {save_path}")
    plt.show()
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot sensor CSV data.")
    parser.add_argument("csv", help="CSV file from sensor_logger.py")
    parser.add_argument("--ranges", default="ranges.txt",
                        help="Pose ranges file (default: ranges.txt)")
    parser.add_argument("--save", help="Optional image file to save, e.g. plot.png")
    args = parser.parse_args()
    plot_sensors(args.csv, args.ranges, args.save)
 