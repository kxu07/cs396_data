"""
Train and evaluate the glove pose classifier.

What it does:
  1. Loads the labeled sensor data in train_sensor/pose*.csv
     (5 flex sensors + pose_id, cut from the time ranges in ranges.txt).
  2. Trims the edges of each pose segment so transition moments are dropped.
  3. Calibrates each session (per-person, per-glove-wearing) so every finger
     reads 0 (most bent) to 1 (most straight).
  4. Smooths the 1000+ Hz stream into short windows and averages each finger,
     so one row = one window = 5 numbers.
  5. Leave-one-person-out test: train on 3 people, test on the 4th, repeat.
     Compares the MLP against a random forest, with and without calibration.
  6. Prints accuracy per held-out person, per-pose accuracy and latency,
     saves confusion matrices, and saves a final MLP trained on everyone
     for the live demo.

Usage:
    python train_glove.py
    python train_glove.py --window-ms 100 --trim-s 0.5 --epochs 300
"""

import argparse
import glob
import json
import os
import re
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

SENSORS = ["thumb", "index", "middle", "ring", "pinky"]
POSE_NAMES = {
    1: "Close Fist",
    2: "Open Palm",
    3: "Point Finger",
    4: "Thumb Up",
    5: "Peace Sign",
    6: "I Love You",
}
SEED = 0


# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------
def load_data(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, "pose*.csv")))
    if not files:
        raise SystemExit(f"No pose*.csv files found in {data_dir}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["session"] = df["source_file"].str.replace(".csv", "", regex=False)
    df["person"] = df["session"].apply(lambda s: re.match(r"[a-zA-Z]+", s).group(0).lower())
    df["label"] = df["pose_id"].astype(int) - 1  # PyTorch wants classes 0..5
    return df


# ---------------------------------------------------------------------------
# 2. Trim transition edges off each (session, pose) segment
# ---------------------------------------------------------------------------
def trim_segments(df, trim_s):
    if trim_s <= 0:
        return df
    keep = []
    for _, seg in df.groupby(["session", "pose_id"]):
        t0, t1 = seg["time_s"].min(), seg["time_s"].max()
        keep.append(seg[(seg["time_s"] >= t0 + trim_s) & (seg["time_s"] <= t1 - trim_s)])
    return pd.concat(keep, ignore_index=True)


# ---------------------------------------------------------------------------
# 3. Per-session calibration
#    Uses the 2nd and 98th percentile of each finger over the whole session,
#    so it needs no labels. In the demo this is a ~10 s "open and close your
#    hand a few times" step before predicting.
# ---------------------------------------------------------------------------
def calibration_ranges(df):
    ranges = {}
    for session, sess in df.groupby("session"):
        lo = sess[SENSORS].quantile(0.02)
        hi = sess[SENSORS].quantile(0.98)
        ranges[session] = (lo, hi)
    return ranges


def apply_calibration(df, ranges):
    out = df.copy()
    out[SENSORS] = out[SENSORS].astype(float)
    for session, (lo, hi) in ranges.items():
        mask = out["session"] == session
        span = (hi - lo).replace(0, 1)
        out.loc[mask, SENSORS] = ((out.loc[mask, SENSORS] - lo) / span).clip(0, 1).values
    return out


# ---------------------------------------------------------------------------
# 4. Windowing: average each finger over short windows
# ---------------------------------------------------------------------------
def make_windows(df, window_ms):
    win_s = window_ms / 1000.0
    rows = []
    for (session, pose_id), seg in df.groupby(["session", "pose_id"]):
        seg = seg.sort_values("time_s")
        win_id = ((seg["time_s"] - seg["time_s"].min()) // win_s).astype(int)
        means = seg.groupby(win_id)[SENSORS].mean()
        counts = seg.groupby(win_id).size()
        means = means[counts >= 0.5 * counts.median()]  # drop a short last window
        means["session"] = session
        means["person"] = seg["person"].iloc[0]
        means["label"] = int(pose_id) - 1
        rows.append(means)
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# 5. Models
# ---------------------------------------------------------------------------
class GloveMLP(nn.Module):
    """5 finger values -> 32 -> 32 -> 6 pose scores."""

    def __init__(self, n_in=5, n_hidden=32, n_out=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, n_hidden), nn.ReLU(),
            nn.Linear(n_hidden, n_hidden), nn.ReLU(),
            nn.Linear(n_hidden, n_out),
        )

    def forward(self, x):
        return self.net(x)


def train_mlp(X, y, epochs, lr=1e-2):
    """Standardize inputs, then train with cross-entropy + Adam (full batch)."""
    torch.manual_seed(SEED)
    mean, std = X.mean(axis=0), X.std(axis=0) + 1e-8
    Xt = torch.tensor((X - mean) / std, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    model = GloveMLP()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(Xt), yt)
        loss.backward()
        opt.step()
    model.eval()
    return model, mean, std


def predict_mlp(model, mean, std, X):
    with torch.no_grad():
        logits = model(torch.tensor((X - mean) / std, dtype=torch.float32))
    return logits.argmax(dim=1).numpy()


def train_rf(X, y):
    rf = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)
    rf.fit(X, y)
    return rf


# ---------------------------------------------------------------------------
# 6. Leave-one-person-out evaluation
# ---------------------------------------------------------------------------
def leave_one_person_out(win, epochs):
    people = sorted(win["person"].unique())
    preds = {"MLP": [], "Random forest": []}
    truth, held = [], []
    per_person = []
    for person in people:
        train, test = win[win["person"] != person], win[win["person"] == person]
        Xtr, ytr = train[SENSORS].values, train["label"].values
        Xte, yte = test[SENSORS].values, test["label"].values

        model, mean, std = train_mlp(Xtr, ytr, epochs)
        p_mlp = predict_mlp(model, mean, std, Xte)
        p_rf = train_rf(Xtr, ytr).predict(Xte)

        preds["MLP"].append(p_mlp)
        preds["Random forest"].append(p_rf)
        truth.append(yte)
        held.append(np.full(len(yte), person))
        per_person.append({
            "held_out": person,
            "windows": len(yte),
            "MLP": (p_mlp == yte).mean(),
            "Random forest": (p_rf == yte).mean(),
        })
    truth = np.concatenate(truth)
    preds = {k: np.concatenate(v) for k, v in preds.items()}
    return pd.DataFrame(per_person), truth, preds


def per_pose_accuracy(truth, pred):
    rows = []
    for c in range(6):
        m = truth == c
        rows.append({"pose": POSE_NAMES[c + 1], "accuracy": (pred[m] == c).mean() if m.any() else np.nan})
    return pd.DataFrame(rows)


def save_confusion(truth, pred, title, path):
    cm = confusion_matrix(truth, pred, labels=range(6), normalize="true")
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(cm, display_labels=[POSE_NAMES[i] for i in range(1, 7)]).plot(
        ax=ax, cmap="Blues", values_format=".2f", colorbar=False)
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def measure_latency_ms(model, mean, std, n=1000):
    x = np.random.rand(1, 5)
    predict_mlp(model, mean, std, x)  # warm-up
    t0 = time.perf_counter()
    for _ in range(n):
        predict_mlp(model, mean, std, x)
    return (time.perf_counter() - t0) / n * 1000


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Train the glove pose classifier.")
    ap.add_argument("--data", default="train_sensor", help="Folder with pose*.csv")
    ap.add_argument("--out", default="results", help="Folder for plots and the saved model")
    ap.add_argument("--window-ms", type=float, default=100, help="Smoothing window length")
    ap.add_argument("--trim-s", type=float, default=0.5, help="Seconds cut from each segment edge")
    ap.add_argument("--epochs", type=int, default=300)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    np.random.seed(SEED)

    df = load_data(args.data)
    print(f"Loaded {len(df):,} raw samples from {df['session'].nunique()} sessions, "
          f"people: {', '.join(sorted(df['person'].unique()))}")

    df = trim_segments(df, args.trim_s)
    ranges = calibration_ranges(df)
    versions = {
        "raw": make_windows(df, args.window_ms),
        "calibrated": make_windows(apply_calibration(df, ranges), args.window_ms),
    }
    print(f"{len(versions['raw']):,} windows of {args.window_ms:.0f} ms "
          f"(trimmed {args.trim_s}s off each segment edge)\n")

    summary = []
    for name, win in versions.items():
        per_person, truth, preds = leave_one_person_out(win, args.epochs)
        print(f"=== Leave-one-person-out, {name} inputs ===")
        print(per_person.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        for model_name, pred in preds.items():
            acc = (pred == truth).mean()
            summary.append({"inputs": name, "model": model_name, "accuracy": acc})
            slug = f"{name}_{model_name.lower().replace(' ', '_')}"
            save_confusion(truth, pred, f"{model_name}, {name} inputs (acc {acc:.2f})",
                           os.path.join(args.out, f"confusion_{slug}.png"))
        print("\nPer-pose accuracy (MLP):")
        print(per_pose_accuracy(truth, preds["MLP"]).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        print()

    summary = pd.DataFrame(summary)
    print("=== Overall accuracy (all held-out people pooled) ===")
    print(summary.pivot(index="model", columns="inputs", values="accuracy")
          .to_string(float_format=lambda v: f"{v:.3f}"))

    # Final model for the demo: MLP on calibrated inputs, trained on everyone.
    win = versions["calibrated"]
    model, mean, std = train_mlp(win[SENSORS].values, win["label"].values, args.epochs)
    latency = measure_latency_ms(model, mean, std)
    print(f"\nMLP latency: {latency:.3f} ms per prediction")

    torch.save(model.state_dict(), os.path.join(args.out, "glove_mlp.pt"))
    with open(os.path.join(args.out, "glove_mlp_config.json"), "w") as f:
        json.dump({
            "sensors": SENSORS,
            "pose_names": [POSE_NAMES[i] for i in range(1, 7)],
            "window_ms": args.window_ms,
            "input_mean": mean.tolist(),
            "input_std": std.tolist(),
            "calibration": "per session: (x - p2) / (p98 - p2), clipped to 0..1",
        }, f, indent=2)
    print(f"Saved demo model and config to {args.out}/, confusion matrices too.")


if __name__ == "__main__":
    main()
