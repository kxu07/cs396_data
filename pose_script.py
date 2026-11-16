import csv
import time
from datetime import datetime
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 1. Configure the Gesture Recognizer
base_options = python.BaseOptions(model_asset_path="gesture_recognizer.task")
options = vision.GestureRecognizerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# 2. Set up CSV file and header columns
csv_filename = f"gesture_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
csv_headers = [
    "iso_timestamp",
    "elapsed_ms",
    "hand_index",
    "handedness",
    "gesture",
    "gesture_confidence",
]
# Append x, y, z columns for all 21 hand landmarks (lm0_x ... lm20_z)
for i in range(21):
    csv_headers.extend([f"lm{i}_x", f"lm{i}_y", f"lm{i}_z"])

# Helper to draw hand skeleton and gesture label
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),         # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),         # Index
    (5, 9), (9, 10), (10, 11), (11, 12),    # Middle
    (9, 13), (13, 14), (14, 15), (15, 16),  # Ring
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) # Pinky & palm
]

def draw_results(frame, result):
    if not result.hand_landmarks:
        return frame
    h, w, _ = frame.shape
    
    for idx, landmarks in enumerate(result.hand_landmarks):
        for start_idx, end_idx in HAND_CONNECTIONS:
            p1 = (int(landmarks[start_idx].x * w), int(landmarks[start_idx].y * h))
            p2 = (int(landmarks[end_idx].x * w), int(landmarks[end_idx].y * h))
            cv2.line(frame, p1, p2, (0, 255, 0), 2)
            
        for lm in landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

        if result.gestures and idx < len(result.gestures):
            gesture = result.gestures[idx][0].category_name
            score = result.gestures[idx][0].score
            wrist_x = int(landmarks[0].x * w)
            wrist_y = int(landmarks[0].y * h) - 20
            cv2.putText(
                frame,
                f"{gesture} ({score:.2f})",
                (wrist_x, max(30, wrist_y)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2,
            )
    return frame

# 3. Start Webcam Capture and Open CSV Writer
cap = cv2.VideoCapture(0)
start_time = time.time()

with open(csv_filename, mode="w", newline="") as csv_file, \
     vision.GestureRecognizer.create_from_options(options) as recognizer:
    
    writer = csv.writer(csv_file)
    writer.writerow(csv_headers)
    print(f"Recording predictions to {csv_filename}...")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Convert BGR (OpenCV) to RGB and wrap in mp.Image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Timestamps: monotonic ms for MediaPipe + wall-clock ISO timestamp for CSV
        elapsed_ms = int((time.time() - start_time) * 1000)
        iso_now = datetime.now().isoformat(timespec="milliseconds")

        # Run gesture recognition
        result = recognizer.recognize_for_video(mp_image, elapsed_ms)

        # Write predictions to CSV if any hand is detected
        if result.hand_landmarks:
            for idx, landmarks in enumerate(result.hand_landmarks):
                handedness = (
                    result.handedness[idx][0].category_name
                    if result.handedness and idx < len(result.handedness)
                    else "Unknown"
                )
                gesture = (
                    result.gestures[idx][0].category_name
                    if result.gestures and idx < len(result.gestures)
                    else "None"
                )
                confidence = (
                    round(result.gestures[idx][0].score, 4)
                    if result.gestures and idx < len(result.gestures)
                    else 0.0
                )

                row = [iso_now, elapsed_ms, idx, handedness, gesture, confidence]
                for lm in landmarks:
                    row.extend([round(lm.x, 6), round(lm.y, 6), round(lm.z, 6)])
                writer.writerow(row)

        # Draw landmarks + gesture prediction on the frame
        annotated_frame = draw_results(frame, result)
        cv2.imshow("MediaPipe Gesture Recorder (Press ESC to quit)", annotated_frame)

        if cv2.waitKey(5) & 0xFF == 27:  # Press ESC to exit
            break

cap.release()
cv2.destroyAllWindows()
print(f"Saved CSV log to {csv_filename}")