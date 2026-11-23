
"""
Predict the hand pose over time from a raw glove recording.
 
Takes a CSV in the sensor_logger.py format
(time_s,index,middle,ring,pinky,thumb) and the model saved by train_glove.py,
and predicts a pose for every window of time (100 ms by default, from the
model config).
 
The input is processed the same way as in training:
  1. Calibrate each finger to 0 (most bent) .. 1 (most straight) using the
     2nd and 98th percentile of this recording.
  2. Average each finger over fixed-length windows.
  3. Standardize with the training mean/std and run the MLP.
 
Usage:
    python predict_pose.py new_person.csv
    python predict_pose.py new_person.csv --model-dir results --plot
    python predict_pose.py new_person.csv --per-sample --min-conf 0.6
    python predict_pose.py new_person.csv --plot --ranges test_ranges.txt
 
Outputs (in ./predictions unless --out is given):
    <name>_predictions.csv          one row per window
    <name>_predictions_samples.csv  every original row + its prediction (--per-sample)
    <name>_predictions.png          sensor plot shaded by predicted pose (--plot)
"""
 
import argparse
import json
import re
from pathlib import Path
 
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
 
 
class GloveMLP(nn.Module):
    """Same architecture as train_glove.py: 5 -> 32 -> 32 -> 6."""
 
    def __init__(self, n_in=5, n_hidden=32, n_out=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, n_hidden), nn.ReLU(),
            nn.Linear(n_hidden, n_hidden), nn.ReLU(),
            nn.Linear(n_hidden, n_out),
        )
 
    def forward(self, x):
        return self.net(x)
 
 
def load_model(model_dir):
    model_dir = Path(model_dir)
    with open(model_dir / "glove_mlp_config.json") as f:
        cfg = json.load(f)
    model = GloveMLP(n_in=len(cfg["sensors"]), n_out=len(cfg["pose_names"]))
    model.load_state_dict(torch.load(model_dir / "glove_mlp.pt", map_location="cpu"))
    model.eval()
    return model, cfg
 
 
def calibrate(df, sensors):
    """Per-recording calibration, as in train_glove.apply_calibration."""
    lo = df[sensors].quantile(0.02)
    hi = df[sensors].quantile(0.98)
    span = (hi - lo).replace(0, 1)
    out = df.copy()
    out[sensors] = ((df[sensors].astype(float) - lo) / span).clip(0, 1)
    return out
 
 
def make_windows(df, sensors, window_ms):
    """Average each finger over consecutive windows of window_ms."""
    win_s = window_ms / 1000.0
    t0 = df["time_s"].min()
    win_id = ((df["time_s"] - t0) // win_s).astype(int)
    g = df.groupby(win_id)
    win = g[sensors].mean()
    win.insert(0, "window_start_s", t0 + win.index * win_s)
    win.insert(1, "window_end_s", win["window_start_s"] + win_s)
    win["n_samples"] = g.size()
    return win.reset_index(names="window"), win_id
 
 
def predict(model, cfg, X):
    mean = np.array(cfg["input_mean"])
    std = np.array(cfg["input_std"])
    with torch.no_grad():
        logits = model(torch.tensor((X - mean) / std, dtype=torch.float32))
        probs = torch.softmax(logits, dim=1).numpy()
    return probs
 
 
def pose_runs(win):
    """Collapse consecutive windows with the same prediction into runs."""
    change = (win["pose_name"] != win["pose_name"].shift()).cumsum()
    return (win.groupby(change)
               .agg(start_s=("window_start_s", "first"),
                    end_s=("window_end_s", "last"),
                    pose=("pose_name", "first"),
                    mean_conf=("confidence", "mean"))
               .reset_index(drop=True))
 
 
LINE_RE = re.compile(r"^\s*(\d+)\s+(.+?)\s*:\s*([\d.]+)\s*-\s*([\d.]+)\s*$")
 
 
def load_true_ranges(ranges_path, csv_name):
    """Return [(pose_id, start_s, end_s), ...] for csv_name from a ranges.txt-style file.
 
    Only the pose number is used, so misspelled names don't matter.
    """
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
            pid, _, start, end = m.groups()
            found.append((int(pid), float(start), float(end)))
    return found
 
 
def true_pose_per_window(win, true_ranges):
    """True pose_id for each window (by its midpoint), 0 if outside every range."""
    mid = (win["window_start_s"] + win["window_end_s"]) / 2
    truth = np.zeros(len(win), dtype=int)
    for pid, start, end in true_ranges:
        truth[((mid >= start) & (mid <= end)).values] = pid
    return truth
 
 
def plot_predictions(df, win, sensors, pose_names, title, save_path, true_ranges=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
 
    colors = plt.get_cmap("Set2").colors
    pose_color = {name: colors[i % len(colors)] for i, name in enumerate(pose_names)}
    pose_color["Uncertain"] = (0.85, 0.85, 0.85)
    t_min, t_max = df["time_s"].min(), df["time_s"].max()
 
    # Two thin timeline bars ("True", "Predicted") above the sensor plot.
    bars = (["True"] if true_ranges else []) + ["Predicted"]
    fig, axes = plt.subplots(
        len(bars) + 1, 1, sharex=True, figsize=(12, 5.5 + 0.4 * len(bars)),
        gridspec_kw={"height_ratios": [0.35] * len(bars) + [5], "hspace": 0.08})
    bar_axes = dict(zip(bars, axes[:-1]))
    ax = axes[-1]
 
    for b, bax in bar_axes.items():
        bax.set_yticks([])
        bax.set_ylabel(b, rotation=0, ha="right", va="center", fontsize=9)
        bax.set_ylim(0, 1)
        for side in ("top", "right", "left"):
            bax.spines[side].set_visible(False)
        bax.tick_params(axis="x", length=0)
 
    if true_ranges:
        tax = bar_axes["True"]
        for pid, start, end in true_ranges:
            name = pose_names[pid - 1]
            tax.axvspan(start, end, color=pose_color[name], linewidth=0)
            tax.text((start + end) / 2, 0.5, name, ha="center", va="center", fontsize=8)
            # Dashed lines on the sensor plot at the true pose boundaries.
            for t in (start, end):
                ax.axvline(t, color="gray", linestyle="--", linewidth=0.7)
 
    pax = bar_axes["Predicted"]
    for _, run in pose_runs(win).iterrows():
        pax.axvspan(run["start_s"], run["end_s"], color=pose_color[run["pose"]],
                    linewidth=0)
 
    for col in sensors:
        ax.plot(df["time_s"], df[col], label=col, linewidth=1)
 
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Reading (0–1023)")
    ax.set_ylim(0, 1023)
    ax.set_xlim(t_min, t_max)
    ax.grid(alpha=0.3)
    axes[0].set_title(title)
 
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(sensors), frameon=False)
    fig.subplots_adjust(bottom=0.16)
    used = set(win["pose_name"])
    if true_ranges:
        used |= {pose_names[pid - 1] for pid, _, _ in true_ranges}
    shown = [p for p in list(pose_names) + ["Uncertain"] if p in used]
    ax.legend(handles=[Patch(color=pose_color[p], label=p) for p in shown],
              title="Pose", loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
 
 
def main():
    ap = argparse.ArgumentParser(description="Predict glove poses over time.")
    ap.add_argument("csv", help="Raw recording (time_s,index,middle,ring,pinky,thumb)")
    ap.add_argument("--model-dir", default="results",
                    help="Folder with glove_mlp.pt and glove_mlp_config.json")
    ap.add_argument("--out", default="predictions",
                    help="Output folder (default: ./predictions)")
    ap.add_argument("--window-ms", type=float,
                    help="Window length (default: the value the model was trained with)")
    ap.add_argument("--min-conf", type=float, default=0.0,
                    help="Label windows below this confidence as 'Uncertain' (0..1)")
    ap.add_argument("--per-sample", action="store_true",
                    help="Also write every original row with its window's prediction")
    ap.add_argument("--plot", action="store_true", help="Save a plot of the predictions")
    ap.add_argument("--ranges",
                    help="Optional ranges.txt-style file with the true pose times; "
                         "the section matching this CSV's file name is used")
    args = ap.parse_args()
 
    model, cfg = load_model(args.model_dir)
    sensors = cfg["sensors"]
    pose_names = cfg["pose_names"]
    window_ms = args.window_ms or cfg.get("window_ms", 100)
 
    csv_path = Path(args.csv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
 
    raw = pd.read_csv(csv_path)
    missing = [c for c in ["time_s"] + sensors if c not in raw.columns]
    if missing:
        raise SystemExit(f"{csv_path} is missing columns: {missing}")
    raw = raw.sort_values("time_s").reset_index(drop=True)
 
    win, win_id = make_windows(calibrate(raw, sensors), sensors, window_ms)
    probs = predict(model, cfg, win[sensors].values)
 
    best = probs.argmax(axis=1)
    win["confidence"] = probs.max(axis=1)
    win["pose_id"] = best + 1
    win["pose_name"] = [pose_names[i] for i in best]
    low = win["confidence"] < args.min_conf
    win.loc[low, "pose_id"] = 0
    win.loc[low, "pose_name"] = "Uncertain"
    for i, name in enumerate(pose_names):
        win[f"p_{name.replace(' ', '_')}"] = probs[:, i]
 
    true_ranges = []
    if args.ranges:
        true_ranges = load_true_ranges(args.ranges, csv_path.name)
        if not true_ranges:
            print(f"WARNING: no [{csv_path.name}] section in {args.ranges}; "
                  "plotting without true poses.")
    if true_ranges:
        win["true_pose_id"] = true_pose_per_window(win, true_ranges)
        win["true_pose_name"] = [pose_names[i - 1] if i else "" for i in win["true_pose_id"]]
 
    cols = (["window_start_s", "window_end_s", "pose_id", "pose_name", "confidence"]
            + (["true_pose_id", "true_pose_name"] if true_ranges else [])
            + [c for c in win.columns if c.startswith("p_")] + ["n_samples"])
    out_csv = out_dir / f"{csv_path.stem}_predictions.csv"
    win[cols].to_csv(out_csv, index=False, float_format="%.4f")
    print(f"Saved {len(win)} window predictions ({window_ms:.0f} ms each) to {out_csv}")
 
    if args.per_sample:
        per = raw.copy()
        mapped = win.set_index("window").loc[win_id.values]
        for c in ["pose_id", "pose_name", "confidence"] + (
                ["true_pose_id", "true_pose_name"] if true_ranges else []):
            per[c] = mapped[c].values
        out_s = out_dir / f"{csv_path.stem}_predictions_samples.csv"
        per.to_csv(out_s, index=False)
        print(f"Saved per-sample predictions to {out_s}")
 
    if args.plot:
        out_png = out_dir / f"{csv_path.stem}_predictions.png"
        plot_predictions(raw, win, sensors, pose_names,
                         f"Predicted poses — {csv_path.name}", out_png, true_ranges)
        print(f"Saved plot to {out_png}")
 
    # Short summary: stretches of the same predicted pose lasting at least 0.5 s.
    runs = pose_runs(win)
    runs = runs[runs["end_s"] - runs["start_s"] >= 0.5]
    print("\nPredicted pose stretches (>= 0.5 s):")
    for _, r in runs.iterrows():
        print(f"  {r['start_s']:7.2f} - {r['end_s']:7.2f} s  {r['pose']:<13} "
              f"(avg conf {r['mean_conf']:.2f})")
 
    # Accuracy over windows that fall inside a labelled range.
    if true_ranges:
        lab = win[win["true_pose_id"] > 0]
        correct = lab["pose_id"] == lab["true_pose_id"]
        print(f"\nAccuracy inside labelled ranges: {correct.mean():.3f} "
              f"({correct.sum()}/{len(lab)} windows)")
        for pid, grp in lab.groupby("true_pose_id"):
            acc = (grp["pose_id"] == pid).mean()
            print(f"  {pose_names[pid - 1]:<13} {acc:.3f}")
 
 
if __name__ == "__main__":
    main()
 