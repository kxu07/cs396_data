
"""
Plot every sensor column from a combined pose CSV (from combine_poses.py)
on one graph, in file order, ignoring the timestamps.
 
Usage:
    python plot_sensors.py combined/pose1_Close_Fist.csv
    python plot_sensors.py combined/pose1_Close_Fist.csv --save fist.png
"""
 
import argparse
 
import matplotlib.pyplot as plt
import pandas as pd
 
NON_SENSOR_COLS = {"time_s", "source_file", "pose_id"}
 
 
def plot_sensors(csv_path, save_path=None):
    """Plot all sensor columns against sample number (row order)."""
    df = pd.read_csv(csv_path)
    sensors = [col for col in df.columns if col not in NON_SENSOR_COLS]
    x = range(len(df))
 
    fig, ax = plt.subplots(figsize=(10, 5))
    for col in sensors:
        ax.plot(x, df[col], label=col, linewidth=1)
 
    # Label each run of consecutive rows from the same source file,
    # with a faint divider line between sections.
    if "source_file" in df.columns:
        section_id = (df["source_file"] != df["source_file"].shift()).cumsum()
        for _, sec in df.groupby(section_id, sort=False):
            start, end = sec.index[0], sec.index[-1]
            if start > 0:
                ax.axvline(start, color="gray", linestyle="--", linewidth=0.8)
            ax.text((start + end) / 2, 0.97, sec["source_file"].iloc[0],
                    transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor="gray", alpha=0.85))
 
    ax.set_xlabel("Sample")
    ax.set_ylabel("Reading (0–1023)")
    ax.set_title(f"Sensor readings — {csv_path}")
    ax.set_ylim(0, 1023)
    ax.set_xlim(0, len(df) - 1)
    ax.grid(alpha=0.3)
    # Legend below the plot so it doesn't cover the file labels.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12),
              ncol=len(sensors), frameon=False)
    fig.tight_layout()
 
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved plot to {save_path}")
    plt.show()
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot a combined pose CSV.")
    parser.add_argument("csv", help="Combined pose CSV from combine_poses.py")
    parser.add_argument("--save", help="Optional image file to save, e.g. plot.png")
    args = parser.parse_args()
    plot_sensors(args.csv, args.save)
 