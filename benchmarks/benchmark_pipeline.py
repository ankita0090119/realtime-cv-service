import time
import cv2
from ultralytics import YOLO


MODEL_PATH = "yolo11n.pt"
VIDEO_PATH = "data/videos/CAM1.mp4"

WARMUP_FRAMES = 5
BENCHMARK_FRAMES = 100

IMGSZ = 416
ZONE_X1 = 1200
ZONE_Y1 = 200
ZONE_X2 = 1630
ZONE_Y2 = 880


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
            imgsz=IMGSZ,
            verbose=False
        )

    inference_times = []
    postprocessing_times = []
    end_to_end_times = []

    processed_frames = 0

    print("Running baseline benchmark...")

    for frame in benchmark_frames:

        # ---------------------------------
        # End-to-end timing starts here
        # ---------------------------------
        frame_start = time.perf_counter()

        # ---------------------------------
        # YOLO + ByteTrack
        # ---------------------------------
        inference_start = time.perf_counter()

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            imgsz=IMGSZ,
            verbose=False
        )

        inference_end = time.perf_counter()

        # ---------------------------------
        # Post-processing + zone analytics
        # ---------------------------------
        post_start = time.perf_counter()

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

                inside_zone = (
                    ZONE_X1 <= center_x <= ZONE_X2
                    and ZONE_Y1 <= center_y <= ZONE_Y2
                )

                if inside_zone:
                    pass

        post_end = time.perf_counter()

        frame_end = time.perf_counter()

        inference_times.append(
            (inference_end - inference_start) * 1000
        )

        postprocessing_times.append(
            (post_end - post_start) * 1000
        )

        end_to_end_times.append(
            (frame_end - frame_start) * 1000
        )

        processed_frames += 1

    if processed_frames == 0:
        raise RuntimeError("No frames were processed")

    total_time = sum(end_to_end_times) / 1000

    avg_inference = sum(inference_times) / processed_frames
    avg_postprocessing = sum(postprocessing_times) / processed_frames
    avg_end_to_end = sum(end_to_end_times) / processed_frames

    fps = 1000 / avg_end_to_end

    print()
    print("=== Baseline Benchmark ===")
    print(f"Frames processed:          {processed_frames}")
    print(f"Average YOLO + tracking:   {avg_inference:.2f} ms")
    print(f"Average post-processing:    {avg_postprocessing:.2f} ms")
    print(f"Average end-to-end latency: {avg_end_to_end:.2f} ms")
    print(f"End-to-end FPS:             {fps:.2f}")
    print(f"Measured processing time:   {total_time:.2f} seconds")


if __name__ == "__main__":
    main()