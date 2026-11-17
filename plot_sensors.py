"""
Plot every sensor column from a sensor_logger.py CSV on one graph.

Usage:
    python plot_sensors.py run1.csv
    python plot_sensors.py run1.csv --save run1.png
"""

import argparse

import matplotlib.pyplot as plt
import pandas as pd


def plot_sensors(csv_path, save_path=None):
    """Plot all sensor columns against time on a single set of axes."""
    df = pd.read_csv(csv_path)
    sensors = [col for col in df.columns if col != "time_s"]

    fig, ax = plt.subplots(figsize=(10, 5))
    for col in sensors:
        ax.plot(df["time_s"], df[col], label=col, linewidth=1)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Reading (0–1023)")
    ax.set_title(f"Sensor readings — {csv_path}")
    ax.set_ylim(0, 1023)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", ncol=len(sensors))
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved plot to {save_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot sensor CSV data.")
    parser.add_argument("csv", help="CSV file from sensor_logger.py")
    parser.add_argument("--save", help="Optional image file to save, e.g. plot.png")
    args = parser.parse_args()
    plot_sensors(args.csv, args.save)