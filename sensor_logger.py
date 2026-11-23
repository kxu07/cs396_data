"""
Record sensor data from the Arduino for a fixed length of time and save it to CSV.

Packet format sent by the Arduino (13 bytes):
    255, then 6 values, each sent as two bytes: (value >> 3), (value & 7)
Both bytes of every value are always < 255, so 255 works as a sync marker.

Usage:
    pip install pyserial
    python sensor_logger.py --port /dev/cu.usbmodem101 --seconds 30
    python sensor_logger.py --port /dev/cu.usbmodem101 --seconds 30 --out run1.csv
"""

import argparse
import csv
import time
from datetime import datetime

import serial  # pip install pyserial

SYNC_BYTE = 255
NUM_VALUES = 5
# HEADER = ["time_s", "index", "middle", "ring", "pinky", "gyro_y", "gyro_x"]
HEADER = ["time_s", "index", "middle", "ring", "pinky", "thumb"]


def read_packet(ser):
    """Wait for the sync byte, then read and decode one packet. Returns a list of 6 ints or None."""
    b = ser.read(1)
    if not b or b[0] != SYNC_BYTE:
        return None
    payload = ser.read(NUM_VALUES * 2)
    if len(payload) != NUM_VALUES * 2 or SYNC_BYTE in payload:
        return None  # incomplete or out of sync; skip it
    return [(payload[2 * i] << 3) | payload[2 * i + 1] for i in range(NUM_VALUES)]


def main():
    parser = argparse.ArgumentParser(description="Log Arduino sensor data to CSV.")
    parser.add_argument("--port", required=True, help="Serial port, e.g. COM3 or /dev/ttyACM0")
    parser.add_argument("--seconds", type=float, default=10, help="How long to record")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--out", help="Output CSV file (default: timestamped name)")
    args = parser.parse_args()

    out_file = args.out or datetime.now().strftime("sensor_log_%Y%m%d_%H%M%S.csv")

    with serial.Serial(args.port, args.baud, timeout=1) as ser:
        print("Connecting... (the Arduino resets when the port opens)")
        time.sleep(2)              # let the Arduino reboot and run setup()
        ser.reset_input_buffer()   # throw away anything sent before we started

        # Keep samples in memory while recording, so slow disk/cloud-synced
        # folders can't stall or interrupt the capture.
        rows = []
        print(f"Recording for {args.seconds} s...")
        start = time.perf_counter()
        while True:
            elapsed = time.perf_counter() - start
            if elapsed >= args.seconds:
                break
            values = read_packet(ser)
            if values is not None:
                rows.append([f"{elapsed:.4f}"] + values)

    # Write everything once recording is finished.
    with open(out_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        writer.writerows(rows)

    print(f"Done. Saved {len(rows)} samples to {out_file} (~{len(rows) / args.seconds:.0f} samples/s)")


if __name__ == "__main__":
    main()