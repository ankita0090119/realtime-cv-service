import threading
import time

import cv2

from app.stream.video_reader import VideoReader
from app.stream.pipeline import FrameProducer

from app.detection.detector import Detector

from app.analytics.zone import Zone
from app.analytics.occupancy import OccupancyCounter
from app.analytics.dwell_time import DwellTimeTracker
from app.analytics.metrics import PerformanceMetrics


class CVService:

    def __init__(self):

        # ---------------------------------------------
        # Video source
        # ---------------------------------------------
        self.video = VideoReader(
            "data/videos/CAM1.mp4"
        )

        # ---------------------------------------------
        # Frame producer
        # ---------------------------------------------
        self.producer = FrameProducer(
            video_reader=self.video,
            buffer_size=2
        )

        # ---------------------------------------------
        # YOLO + ByteTrack
        # ---------------------------------------------
        self.detector = Detector()

        # ---------------------------------------------
        # Analytics
        # ---------------------------------------------
        self.occupancy_counter = OccupancyCounter()

        self.dwell_tracker = DwellTimeTracker()

        self.metrics = PerformanceMetrics()

        # ---------------------------------------------
        # Monitoring zone
        # ---------------------------------------------
        self.zone = Zone(
            name="Product Area",
            x1=1200,
            y1=200,
            x2=1630,
            y2=880
        )

        # ---------------------------------------------
        # Processing state
        # ---------------------------------------------
        self.running = False
        self.processing_thread = None

        # ---------------------------------------------
        # Latest metrics
        # ---------------------------------------------
        self.last_metrics = {
            "occupancy": 0,
            "total_entries": 0,
            "average_dwell_seconds": 0.0,
            "fps": 0.0,
            "inference_latency_ms": 0.0,
            "dropped_frames": 0
        }

    # ==================================================
    # START SERVICE
    # ==================================================

    def start(self):

        if self.running:
            return

        self.running = True

        # Start frame capture
        self.producer.start()

        # Start CV processing thread
        self.processing_thread = threading.Thread(
            target=self._processing_loop,
            daemon=True
        )

        self.processing_thread.start()

    # ==================================================
    # BACKGROUND PROCESSING LOOP
    # ==================================================

    def _processing_loop(self):

        while self.running:

            result = self.process_frame()

            # No frame currently available
            if result is None:

                # Video has ended
                if not self.producer.running:
                    break

                time.sleep(0.001)

    # ==================================================
    # PROCESS ONE FRAME
    # ==================================================

    def process_frame(self):

        packet = self.producer.get_frame()

        if packet is None:
            return None

        frame, timestamp = packet

        # ---------------------------------------------
        # YOLO + ByteTrack
        # ---------------------------------------------

        inference_start = (
            self.metrics.start_inference()
        )

        results = self.detector.predict(frame)

        self.metrics.end_inference(
            inference_start
        )

        self.metrics.update()

        result = results[0]

        # ---------------------------------------------
        # Draw detections
        # ---------------------------------------------

        annotated_frame = result.plot()

        # ---------------------------------------------
        # Find people inside zone
        # ---------------------------------------------

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

                center_x = int(
                    (x1 + x2) / 2
                )

                center_y = int(
                    (y1 + y2) / 2
                )

                # Check zone
                if self.zone.contains(
                    center_x,
                    center_y
                ):

                    inside_ids.append(
                        track_id
                    )

                    current_dwell = (
                        self.dwell_tracker
                        .get_current_dwell(
                            track_id,
                            timestamp
                        )
                    )

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

        # ---------------------------------------------
        # Occupancy
        # ---------------------------------------------

        self.occupancy_counter.update(
            inside_ids
        )

        occupancy = (
            self.occupancy_counter.count()
        )

        # ---------------------------------------------
        # Entry / Exit / Dwell
        # ---------------------------------------------

        self.dwell_tracker.update(
            inside_ids,
            timestamp
        )

        # ---------------------------------------------
        # Performance
        # ---------------------------------------------

        fps = self.metrics.get_fps()

        latency = (
            self.metrics
            .get_average_latency_ms()
        )

        dropped = (
            self.producer.dropped_frames()
        )

        # ---------------------------------------------
        # Draw zone
        # ---------------------------------------------

        cv2.rectangle(
            annotated_frame,
            (self.zone.x1, self.zone.y1),
            (self.zone.x2, self.zone.y2),
            (255, 0, 0),
            2
        )

        cv2.putText(
            annotated_frame,
            self.zone.name,
            (
                self.zone.x1,
                self.zone.y1 - 10
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2
        )

        # ---------------------------------------------
        # Store latest metrics
        # ---------------------------------------------

        self.last_metrics = {
            "occupancy": occupancy,
            "total_entries": (
                self.dwell_tracker.total_entries
            ),
            "average_dwell_seconds": round(
                self.dwell_tracker.get_average_dwell(),
                2
            ),
            "fps": round(fps, 2),
            "inference_latency_ms": round(
                latency,
                2
            ),
            "dropped_frames": dropped
        }

        return {
            "frame": annotated_frame,
            "metrics": self.last_metrics
        }

    # ==================================================
    # GET METRICS
    # ==================================================

    def get_metrics(self):

        return self.last_metrics

    # ==================================================
    # STOP SERVICE
    # ==================================================

    def stop(self):

        if not self.running:
            return

        self.running = False

        self.producer.stop()

        if self.processing_thread is not None:

            self.processing_thread.join(
                timeout=2
            )

        self.video.release()