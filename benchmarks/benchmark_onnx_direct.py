import time
import cv2
import numpy as np
import onnxruntime as ort


MODEL_PATH = "yolo11n.onnx"
IMAGE_PATH = "benchmarks/data/test.jpg"

IMAGE_SIZE = 640
WARMUP_RUNS = 5
BENCHMARK_RUNS = 50


def preprocess(image):
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))

    # OpenCV loads images as BGR.
    # YOLO expects RGB.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert HWC -> CHW
    image = image.transpose(2, 0, 1)

    # Convert uint8 -> float32 and normalize to [0, 1]
    image = image.astype(np.float32) / 255.0

    # Add batch dimension
    image = np.expand_dims(image, axis=0)

    return image


def main():
    image = cv2.imread(IMAGE_PATH)

    if image is None:
        raise RuntimeError(f"Could not read image: {IMAGE_PATH}")

    print("Loading ONNX model...")

    session = ort.InferenceSession(
        MODEL_PATH,
        providers=["CPUExecutionProvider"]
    )

    input_name = session.get_inputs()[0].name

    print(f"Input name: {input_name}")
    print(f"Providers: {session.get_providers()}")

    input_tensor = preprocess(image)

    print("Running warmup...")

    for _ in range(WARMUP_RUNS):
        session.run(
            None,
            {input_name: input_tensor}
        )

    print("Running benchmark...")

    start = time.perf_counter()

    for _ in range(BENCHMARK_RUNS):
        session.run(
            None,
            {input_name: input_tensor}
        )

    elapsed = time.perf_counter() - start

    average_latency = elapsed / BENCHMARK_RUNS
    fps = 1 / average_latency

    print()
    print("=== Direct ONNX Runtime Benchmark ===")
    print(f"Runs: {BENCHMARK_RUNS}")
    print(f"Average latency: {average_latency * 1000:.2f} ms")
    print(f"Throughput: {fps:.2f} FPS")


if __name__ == "__main__":
    main()