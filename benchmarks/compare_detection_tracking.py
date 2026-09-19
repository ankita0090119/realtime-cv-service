import time

import cv2
from ultralytics import YOLO


MODEL_PATH = "yolo11n.pt"
VIDEO_PATH = "data/videos/CAM1.mp4"

WARMUP_RUNS = 5
BENCHMARK_RUNS = 30


# =========================================================
# Load model
# =========================================================

model = YOLO(MODEL_PATH)


# =========================================================
# Read one real frame from the CCTV video
# =========================================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(
        f"Could not open video: {VIDEO_PATH}"
    )

success, frame = cap.read()

cap.release()

if not success:
    raise RuntimeError(
        "Could not read frame from video"
    )


print(
    "Benchmark frame:",
    frame.shape[1],
    "x",
    frame.shape[0]
)


# =========================================================
# Warmup
# =========================================================

print("Running warmup...")

for _ in range(WARMUP_RUNS):

    model.predict(
        frame,
        verbose=False
    )


# =========================================================
# Benchmark YOLO detection
# =========================================================

print("Benchmarking YOLO detection...")

predict_times = []

for _ in range(BENCHMARK_RUNS):

    start = time.perf_counter()

    model.predict(
        frame,
        verbose=False
    )

    elapsed = (
        time.perf_counter() - start
    ) * 1000

    predict_times.append(elapsed)


# =========================================================
# Benchmark YOLO + ByteTrack
# =========================================================

print("Benchmarking YOLO + ByteTrack...")

track_times = []

for _ in range(BENCHMARK_RUNS):

    start = time.perf_counter()

    model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False
    )

    elapsed = (
        time.perf_counter() - start
    ) * 1000

    track_times.append(elapsed)


# =========================================================
# Calculate averages
# =========================================================

average_predict = (
    sum(predict_times)
    / len(predict_times)
)

average_track = (
    sum(track_times)
    / len(track_times)
)


# =========================================================
# Results
# =========================================================

print()

print("=" * 50)
print("RESULTS")
print("=" * 50)

print(
    f"YOLO detection: "
    f"{average_predict:.2f} ms"
)

print(
    f"YOLO + ByteTrack: "
    f"{average_track:.2f} ms"
)

print(
    f"Tracking overhead: "
    f"{average_track - average_predict:.2f} ms"
)

print(
    f"Detection FPS: "
    f"{1000 / average_predict:.2f}"
)

print(
    f"Tracking FPS: "
    f"{1000 / average_track:.2f}"
)

print("=" * 50)