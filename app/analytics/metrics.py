import time
from collections import deque


class PerformanceMetrics:
    def __init__(self, window_size=100):
        self.frame_count = 0
        self.start_time = time.perf_counter()

        self.inference_times = deque(maxlen=window_size)
        self.frame_timestamps = deque(maxlen=window_size)

    def start_inference(self):
        return time.perf_counter()

    def end_inference(self, start_time):
        elapsed = time.perf_counter() - start_time
        self.inference_times.append(elapsed)
        return elapsed

    def update(self):
        self.frame_count += 1
        self.frame_timestamps.append(time.perf_counter())

    def get_fps(self):
        # For a single processed frame, report the
        # current elapsed-time FPS estimate.
        if len(self.frame_timestamps) == 1:
            elapsed = time.perf_counter() - self.frame_timestamps[0]

            if elapsed <= 0:
                return 0.0

            return 1.0 / elapsed

        elapsed = (
            self.frame_timestamps[-1]
            - self.frame_timestamps[0]
        )

        if elapsed <= 0:
            return 0.0

        return (len(self.frame_timestamps) - 1) / elapsed

    def get_average_latency_ms(self):
        if not self.inference_times:
            return 0.0

        average_seconds = (
            sum(self.inference_times)
            / len(self.inference_times)
        )

        return average_seconds * 1000