import time


class PerformanceMetrics:

    def __init__(self):
        self.frame_count = 0
        self.start_time = time.perf_counter()
        self.inference_times = []

    def start_inference(self):
        return time.perf_counter()

    def end_inference(self, start_time):
        elapsed = time.perf_counter() - start_time

        self.inference_times.append(elapsed)

        return elapsed

    def update(self):
        self.frame_count += 1

    def get_fps(self):
        elapsed = time.perf_counter() - self.start_time

        if elapsed <= 0:
            return 0.0

        return self.frame_count / elapsed

    def get_average_latency_ms(self):
        if not self.inference_times:
            return 0.0

        average_seconds = (
            sum(self.inference_times)
            / len(self.inference_times)
        )

        return average_seconds * 1000