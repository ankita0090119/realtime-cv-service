import logging
import threading
import time

import cv2

from app.config import settings

from app.stream.video_reader import VideoReader
from app.stream.pipeline import FrameProducer

from app.detection.detector import Detector

from app.analytics.zone import Zone
from app.analytics.occupancy import OccupancyCounter
from app.analytics.dwell_time import DwellTimeTracker
from app.analytics.metrics import PerformanceMetrics


logger = logging.getLogger(__name__)


class CVService:

    def __init__(self):

        self.video = VideoReader(
            settings.video_source
        )

        self.producer = FrameProducer(
            video_reader=self.video,
            buffer_size=settings.buffer_size
        )

        self.detector = Detector(
            settings.model_path
        )

        self.occupancy_counter = OccupancyCounter()
        self.dwell_tracker = DwellTimeTracker()
        self.metrics = PerformanceMetrics()

        self.zone = Zone(
            name=settings.zone_name,
            x1=settings.zone_x1,
            y1=settings.zone_y1,
            x2=settings.zone_x2,
            y2=settings.zone_y2
        )

        self.running = False
        self.processing_thread = None

        self.processing_error = None

        self.latest_frame = None
        self.frame_id = 0
        self.frame_lock = threading.Lock()

        self.last_metrics = {
            "occupancy": 0,
            "total_entries": 0,
            "average_dwell_seconds": 0.0,
            "fps": 0.0,
            "inference_latency_ms": 0.0,
            "dropped_frames": 0
        }

    def start(self):

        if self.running:
            return

        self.running = True

        self.producer.start()

        self.processing_thread = threading.Thread(
            target=self._processing_loop,
            daemon=True
        )

        self.processing_thread.start()

    def _processing_loop(self):

        logger.info(
            "CV processing loop started"
        )

        while self.running:

            try:

                result = self.process_frame()

                if result is None:

                    if not self.producer.running:

                        logger.warning(
                            "Frame producer stopped"
                        )

                        break

                    time.sleep(0.001)

            except Exception as exc:

                self.processing_error = str(exc)

                logger.exception(
                    "Error in CV processing loop"
                )

                self.running = False

                break

        logger.info(
            "CV processing loop stopped"
        )

    def process_frame(self):

        packet = self.producer.get_frame()

        if packet is None:
            return None

        frame, timestamp = packet

        inference_start = (
            self.metrics.start_inference()
        )

        results = self.detector.predict(
            frame
        )

        self.metrics.end_inference(
            inference_start
        )

        self.metrics.update()

        result = results[0]

        annotated_frame = result.plot()

        inside_ids = []

        if result.boxes.id is not None:

            boxes = (
                result.boxes
                .xyxy
                .cpu()
                .numpy()
            )

            track_ids = (
                result.boxes
                .id
                .cpu()
                .numpy()
                .astype(int)
            )

            classes = (
                result.boxes
                .cls
                .cpu()
                .numpy()
                .astype(int)
            )

            for box, track_id, class_id in zip(
                boxes,
                track_ids,
                classes
            ):

                # Only process person class
                if class_id != 0:
                    continue

                x1, y1, x2, y2 = box

                center_x = int(
                    (x1 + x2) / 2
                )

                center_y = int(
                    (y1 + y2) / 2
                )

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

        # Update occupancy
        self.occupancy_counter.update(
            inside_ids
        )

        occupancy = (
            self.occupancy_counter.count()
        )

        # Update entry/exit and dwell time
        self.dwell_tracker.update(
            inside_ids,
            timestamp
        )

        # Performance metrics
        fps = self.metrics.get_fps()

        latency = (
            self.metrics
            .get_average_latency_ms()
        )

        dropped = (
            self.producer
            .dropped_frames()
        )

        # Draw zone
        cv2.rectangle(
            annotated_frame,
            (
                self.zone.x1,
                self.zone.y1
            ),
            (
                self.zone.x2,
                self.zone.y2
            ),
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

        # Store metrics
        self.last_metrics = {
            "occupancy": occupancy,

            "total_entries": (
                self.dwell_tracker
                .total_entries
            ),

            "average_dwell_seconds": round(
                self.dwell_tracker
                .get_average_dwell(),
                2
            ),

            "fps": round(
                fps,
                2
            ),

            "inference_latency_ms": round(
                latency,
                2
            ),

            "dropped_frames": dropped
        }

        # Store latest frame safely
        with self.frame_lock:

            self.latest_frame = (
                annotated_frame.copy()
            )

            self.frame_id += 1

        return {
            "frame": annotated_frame,
            "metrics": self.last_metrics
        }

    def get_metrics(self):

        return self.last_metrics.copy()

    def get_latest_frame(self):

        with self.frame_lock:

            if self.latest_frame is None:
                return None

            return (
                self.latest_frame.copy(),
                self.frame_id
            )

    def get_status(self):

        return {
            "processing": self.running,

            "model_loaded": (
                self.detector.model
                is not None
            ),

            "video_source": (
                self.video.source
            ),

            "video_open": (
                self.video.capture.isOpened()
            ),

            "processing_error": (
                self.processing_error
            )
        }

    def stop(self):

        if not self.running:
            return

        logger.info(
            "Stopping CV processing"
        )

        self.running = False

        self.producer.stop()

        if self.processing_thread is not None:

            self.processing_thread.join(
                timeout=2
            )

        self.video.release()

        logger.info(
            "CV processing stopped"
        )