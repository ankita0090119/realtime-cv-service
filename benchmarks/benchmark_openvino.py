import time
import cv2
from ultralytics import YOLO


MODEL_PATH = "yolo11n_openvino_model"
IMAGE_PATH = "benchmarks/data/test.jpg"

WARMUP_RUNS = 5
BENCHMARK_RUNS = 50


def main():
    image = cv2.imread(IMAGE_PATH)

    if image is None:
        raise RuntimeError(f"Could not read image: {IMAGE_PATH}")

    print("Loading OpenVINO model...")

    model = YOLO(MODEL_PATH)

    print("Running warmup...")

    for _ in range(WARMUP_RUNS):
        model.predict(
            image,
            imgsz=640,
            verbose=False
        )

    print("Running benchmark...")

    start = time.perf_counter()

    for _ in range(BENCHMARK_RUNS):
        model.predict(
            image,
            imgsz=640,
            verbose=False
        )

    elapsed = time.perf_counter() - start

    average_latency = elapsed / BENCHMARK_RUNS
    fps = 1 / average_latency

    print()
    print("=== OpenVINO YOLO Benchmark ===")
    print(f"Runs: {BENCHMARK_RUNS}")
    print(f"Average latency: {average_latency * 1000:.2f} ms")
    print(f"Throughput: {fps:.2f} FPS")


if __name__ == "__main__":
    main()