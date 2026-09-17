import cv2
import time

from stream.video_reader import VideoReader
from stream.pipeline import FrameProducer

from detection.detector import Detector

from analytics.zone import Zone
from analytics.occupancy import OccupancyCounter
from analytics.dwell_time import DwellTimeTracker
from analytics.metrics import PerformanceMetrics


def main():

    # --------------------------------------------------
    # 1. Video source
    # --------------------------------------------------
    video = VideoReader(
        "data/videos/CAM1.mp4"
    )

    # --------------------------------------------------
    # 2. Frame producer + bounded buffer
    # --------------------------------------------------
    producer = FrameProducer(
        video_reader=video,
        buffer_size=2
    )

    # --------------------------------------------------
    # 3. YOLO + ByteTrack
    # --------------------------------------------------
    detector = Detector()

    # --------------------------------------------------
    # 4. Analytics components
    # --------------------------------------------------
    occupancy_counter = OccupancyCounter()
    dwell_tracker = DwellTimeTracker()
    metrics = PerformanceMetrics()

    # --------------------------------------------------
    # 5. Monitoring zone
    # --------------------------------------------------
    zone = Zone(
        name="Product Area",
        x1=1200,
        y1=200,
        x2=1630,
        y2=880
    )

    # --------------------------------------------------
    # 6. Start frame producer
    # --------------------------------------------------
    producer.start()

    try:

        # --------------------------------------------------
        # 7. Main processing loop
        # --------------------------------------------------
        while True:

            # Get frame packet from buffer
            packet = producer.get_frame()

            # No frame currently available
            if packet is None:

                # Producer stopped and buffer is empty
                if not producer.running:
                    break

                time.sleep(0.001)
                continue

            # Extract frame and timestamp
            frame, timestamp = packet

            # --------------------------------------------------
            # 8. YOLO + ByteTrack inference
            # --------------------------------------------------
            inference_start = (
                metrics.start_inference()
            )

            results = detector.predict(frame)

            inference_latency = (
                metrics.end_inference(
                    inference_start
                )
            )

            metrics.update()

            result = results[0]

            # Draw detections and tracking IDs
            annotated_frame = result.plot()

            # --------------------------------------------------
            # 9. Find people inside zone
            # --------------------------------------------------
            inside_ids = []

            if result.boxes.id is not None:

                boxes = (
                    result.boxes.xyxy
                    .cpu()
                    .numpy()
                )

                track_ids = (
                    result.boxes.id
                    .cpu()
                    .numpy()
                    .astype(int)
                )

                classes = (
                    result.boxes.cls
                    .cpu()
                    .numpy()
                    .astype(int)
                )

                for box, track_id, class_id in zip(
                    boxes,
                    track_ids,
                    classes
                ):

                    # COCO class 0 = person
                    if class_id != 0:
                        continue

                    x1, y1, x2, y2 = box

                    # Bounding-box center
                    center_x = int(
                        (x1 + x2) / 2
                    )

                    center_y = int(
                        (y1 + y2) / 2
                    )

                    # --------------------------------------------------
                    # 10. Zone check
                    # --------------------------------------------------
                    if zone.contains(
                        center_x,
                        center_y
                    ):

                        inside_ids.append(
                            track_id
                        )

                        # Current dwell time
                        current_dwell = (
                            dwell_tracker
                            .get_current_dwell(
                                track_id,
                                timestamp
                            )
                        )

                        # Display dwell time
                        cv2.putText(
                            annotated_frame,
                            f"{current_dwell:.1f}s",
                            (
                                center_x,
                                center_y
                            ),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 255),
                            2
                        )

            # --------------------------------------------------
            # 11. Occupancy
            # --------------------------------------------------
            occupancy_counter.update(
                inside_ids
            )

            occupancy = (
                occupancy_counter.count()
            )

            # --------------------------------------------------
            # 12. Entry / Exit / Dwell
            # --------------------------------------------------
            entered_ids, exited_ids = (
                dwell_tracker.update(
                    inside_ids,
                    timestamp
                )
            )

            # --------------------------------------------------
            # 13. Performance metrics
            # --------------------------------------------------
            fps = metrics.get_fps()

            average_latency = (
                metrics
                .get_average_latency_ms()
            )

            dropped_frames = (
                producer.dropped_frames()
            )

            # --------------------------------------------------
            # 14. Draw monitoring zone
            # --------------------------------------------------
            cv2.rectangle(
                annotated_frame,
                (zone.x1, zone.y1),
                (zone.x2, zone.y2),
                (255, 0, 0),
                2
            )

            cv2.putText(
                annotated_frame,
                zone.name,
                (
                    zone.x1,
                    zone.y1 - 10
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2
            )

            # --------------------------------------------------
            # 15. Display occupancy
            # --------------------------------------------------
            cv2.putText(
                annotated_frame,
                f"Occupancy: {occupancy}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            # --------------------------------------------------
            # 16. Display total entries
            # --------------------------------------------------
            cv2.putText(
                annotated_frame,
                f"Total Entries: "
                f"{dwell_tracker.total_entries}",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

            # --------------------------------------------------
            # 17. Display average dwell
            # --------------------------------------------------
            cv2.putText(
                annotated_frame,
                f"Avg Dwell: "
                f"{dwell_tracker.get_average_dwell():.1f}s",
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

            # --------------------------------------------------
            # 18. Display FPS
            # --------------------------------------------------
            cv2.putText(
                annotated_frame,
                f"FPS: {fps:.1f}",
                (20, 145),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2
            )

            # --------------------------------------------------
            # 19. Display inference latency
            # --------------------------------------------------
            cv2.putText(
                annotated_frame,
                f"Inference: "
                f"{average_latency:.1f} ms",
                (20, 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2
            )

            # --------------------------------------------------
            # 20. Display dropped frames
            # --------------------------------------------------
            cv2.putText(
                annotated_frame,
                f"Dropped Frames: "
                f"{dropped_frames}",
                (20, 215),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 165, 255),
                2
            )

            # --------------------------------------------------
            # 21. Display frame
            # --------------------------------------------------
            cv2.imshow(
                "Real-Time Detection",
                annotated_frame
            )

            # Press Q to exit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:

        # --------------------------------------------------
        # 22. Graceful shutdown
        # --------------------------------------------------
        producer.stop()

        video.release()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()