import cv2
import time


class VideoReader:

    def __init__(self, source):

        self.source = source

        self.capture = cv2.VideoCapture(
            source
        )

        if not self.capture.isOpened():
            raise RuntimeError(
                f"Could not open video source: {source}"
            )

        # Get source FPS
        self.fps = (
            self.capture.get(
                cv2.CAP_PROP_FPS
            )
        )

        if self.fps <= 0:
            self.fps = 25.0

        # Time between frames
        self.frame_interval = (
            1.0 / self.fps
        )

        self.last_frame_time = time.perf_counter()

    def read(self):

        # Maintain approximately the
        # original video frame rate.
        now = time.perf_counter()

        elapsed = (
            now - self.last_frame_time
        )

        if elapsed < self.frame_interval:

            time.sleep(
                self.frame_interval - elapsed
            )

        self.last_frame_time = (
            time.perf_counter()
        )

        success, frame = (
            self.capture.read()
        )

        if not success:

            # Restart video when EOF is reached
            self.capture.set(
                cv2.CAP_PROP_POS_FRAMES,
                0
            )

            success, frame = (
                self.capture.read()
            )

            if not success:
                return None

            self.last_frame_time = (
                time.perf_counter()
            )

        return frame

    def get_timestamp(self):

        return (
            self.capture.get(
                cv2.CAP_PROP_POS_MSEC
            ) / 1000.0
        )

    def release(self):

        self.capture.release()