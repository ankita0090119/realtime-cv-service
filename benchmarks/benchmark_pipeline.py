import time
import cv2
from ultralytics import YOLO


MODEL_PATH = "yolo11n.pt"
VIDEO_PATH = "data/videos/CAM1.mp4"

WARMUP_FRAMES = 5
BENCHMARK_FRAMES = 50


def main():
    model = YOLO(MODEL_PATH)

    capture = cv2.VideoCapture(VIDEO_PATH)

    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    frames = []

    print("Reading video frames...")

    for _ in range(WARMUP_FRAMES + BENCHMARK_FRAMES):
        success, frame = capture.read()

        if not success:
            break

        frames.append(frame)

    capture.release()

    if len(frames) <= WARMUP_FRAMES:
        raise RuntimeError("Not enough frames available for benchmark")

    warmup_frames = frames[:WARMUP_FRAMES]
    benchmark_frames = frames[WARMUP_FRAMES:]

    print("Running warmup...")

    for frame in warmup_frames:
        model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False
        )

    print("Running service pipeline benchmark...")

    start = time.perf_counter()

    processed_frames = 0

    for frame in benchmark_frames:
        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False
        )

        # Simulate the basic post-processing
        # performed by our CV service.
        result = results[0]

        if result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            track_ids = result.boxes.id.cpu().numpy().astype(int)
            classes = result.boxes.cls.cpu().numpy().astype(int)

            for box, track_id, class_id in zip(
                boxes,
                track_ids,
                classes
            ):
                if class_id != 0:
                    continue

                x1, y1, x2, y2 = box

                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                # Simulate zone check.
                inside_zone = (
                    1200 <= center_x <= 1630
                    and 200 <= center_y <= 880
                )

                if inside_zone:
                    pass

        processed_frames += 1

    elapsed = time.perf_counter() - start

    if processed_frames == 0:
        raise RuntimeError("No frames were processed")

    average_latency = elapsed / processed_frames
    fps = processed_frames / elapsed

    print()
    print("=== Service Pipeline Benchmark ===")
    print(f"Frames processed: {processed_frames}")
    print(f"Total time: {elapsed:.2f} seconds")
    print(f"Average processing latency: {average_latency * 1000:.2f} ms")
    print(f"Throughput: {fps:.2f} FPS")


if __name__ == "__main__":
    main()