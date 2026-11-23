#!/usr/bin/env python3
"""
Combine hand-pose sensor CSVs into one CSV per pose.

Usage:
    python combine_poses.py <data_folder> <ranges_file> [output_folder]

ranges file format (paste each file's list under its file name):

    [jay1.csv]
    5 Peace Sign: 9.85 - 19.33
    3 Point Finger: 21.37 - 28.22
    4 Thumb UP: 29.53 - 38.04
    1 Close Fist: 39.70 - 49.35
    6 I love You: 51.13 - 59.60
    2 Open Palm: 60.62 - 70

    [jay2.csv]
    ...

Each output file (e.g. pose1_Close_Fist.csv) has the original sensor columns
plus 'source_file' and 'pose_id', with rows in file order, then time order.
"""
import csv
import re
import sys
from pathlib import Path

LINE_RE = re.compile(r"^\s*(\d+)\s+(.+?)\s*:\s*([\d.]+)\s*-\s*([\d.]+)\s*$")


def parse_ranges(path):
    """Return {filename: [(pose_id, pose_name, start, end), ...]}."""
    ranges, current = {}, None
    for n, line in enumerate(Path(path).read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1].strip()
            ranges[current] = []
            continue
        m = LINE_RE.match(line)
        if not m or current is None:
            sys.exit(f"{path}, line {n}: can't read '{line}'")
        pid, name, start, end = m.groups()
        ranges[current].append((int(pid), name, float(start), float(end)))
    return ranges


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    data_dir = Path(sys.argv[1])
    ranges = parse_ranges(sys.argv[2])
    out_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else data_dir / "combined"
    out_dir.mkdir(parents=True, exist_ok=True)

    # pose_id -> {"name": str, "rows": list}
    poses = {}
    header = None

    for fname, file_ranges in ranges.items():
        fpath = data_dir / fname
        if not fpath.exists():
            print(f"WARNING: {fpath} not found, skipping")
            continue
        with open(fpath, newline="") as f:
            reader = csv.reader(f)
            file_header = next(reader)
            if header is None:
                header = file_header
            elif file_header != header:
                sys.exit(f"{fname} has different columns: {file_header}")
            rows = [r for r in reader if r]

        for pid, name, start, end in file_ranges:
            entry = poses.setdefault(pid, {"name": name, "rows": []})
            picked = [r + [fname, pid] for r in rows
                      if start <= float(r[0]) <= end]
            entry["rows"].extend(picked)
            print(f"{fname}: pose {pid} {name} ({start}-{end}s): {len(picked)} rows")

    for pid in sorted(poses):
        name = poses[pid]["name"]
        safe = re.sub(r"\W+", "_", name).strip("_")
        out = out_dir / f"pose{pid}_{safe}.csv"
        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header + ["source_file", "pose_id"])
            w.writerows(poses[pid]["rows"])
        print(f"Wrote {out} ({len(poses[pid]['rows'])} rows)")


if __name__ == "__main__":
    main()