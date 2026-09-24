import time
import cv2
import torch
from ultralytics import YOLO

MODEL_PATH = "yolo11n.pt"
VIDEO_PATH = "data/videos/CAM1.mp4"

IMGSZ = 416
WARMUP = 5
RUNS = 50


def load_frames():
    cap = cv2.VideoCapture(VIDEO_PATH)
    frames = []

    while len(frames) < WARMUP + RUNS:
        success, frame = cap.read()
        if not success:
            break
        frames.append(frame)

    cap.release()
    return frames


def benchmark(model, frames, half):
    for frame in frames[:WARMUP]:
        model.predict(
            frame,
            imgsz=IMGSZ,
            device="cpu",
            half=half,
            verbose=False
        )

    start = time.perf_counter()

    for frame in frames[WARMUP:]:
        model.predict(
            frame,
            imgsz=IMGSZ,
            device="cpu",
            half=half,
            verbose=False
        )

    elapsed = time.perf_counter() - start

    fps = RUNS / elapsed
    latency = (elapsed / RUNS) * 1000

    return latency, fps


def main():
    print("PyTorch:", torch.__version__)
    print("MKLDNN:", torch.backends.mkldnn.is_available())

    model = YOLO(MODEL_PATH)
    frames = load_frames()

    print("\n=== FP16 CPU Benchmark ===")

    fp32_latency, fp32_fps = benchmark(model, frames, False)

    print(
        f"FP32: {fp32_latency:.2f} ms/frame | "
        f"{fp32_fps:.2f} FPS"
    )

    fp16_latency, fp16_fps = benchmark(model, frames, True)

    print(
        f"FP16: {fp16_latency:.2f} ms/frame | "
        f"{fp16_fps:.2f} FPS"
    )


if __name__ == "__main__":
    main()