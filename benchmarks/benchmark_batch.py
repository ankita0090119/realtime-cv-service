import time
import cv2
from ultralytics import YOLO

VIDEO_PATH = "data/videos/CAM1.mp4"
MODEL_PATH = "yolo11n.pt"

BATCH_SIZES = [1, 2, 4, 8]
IMGSZ = 416
WARMUP_BATCHES = 2
BENCHMARK_BATCHES = 10


def load_frames():
    cap = cv2.VideoCapture(VIDEO_PATH)
    frames = []

    while len(frames) < 100:
        success, frame = cap.read()
        if not success:
            break
        frames.append(frame)

    cap.release()
    return frames


def benchmark(model, frames, batch_size):
    batches = [
        frames[i:i + batch_size]
        for i in range(0, len(frames), batch_size)
    ]

    batches = [b for b in batches if len(b) == batch_size]

    for batch in batches[:WARMUP_BATCHES]:
        model.predict(
            batch,
            imgsz=IMGSZ,
            device="cpu",
            verbose=False
        )

    total_frames = 0
    start = time.perf_counter()

    for batch in batches[:BENCHMARK_BATCHES]:
        model.predict(
            batch,
            imgsz=IMGSZ,
            device="cpu",
            verbose=False
        )
        total_frames += len(batch)

    elapsed = time.perf_counter() - start

    fps = total_frames / elapsed
    latency = (elapsed / total_frames) * 1000

    return latency, fps


def main():
    print("Loading model...")
    model = YOLO(MODEL_PATH)

    print("Loading video...")
    frames = load_frames()

    print(f"Loaded {len(frames)} frames\n")
    print("=== Batch Inference Benchmark ===")

    for batch_size in BATCH_SIZES:
        latency, fps = benchmark(model, frames, batch_size)

        print(
            f"Batch {batch_size}: "
            f"{latency:.2f} ms/frame | "
            f"{fps:.2f} FPS"
        )


if __name__ == "__main__":
    main()