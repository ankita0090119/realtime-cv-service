import time

from app.analytics.metrics import PerformanceMetrics


def test_frame_count_increases():

    metrics = PerformanceMetrics()

    metrics.update()
    metrics.update()
    metrics.update()

    assert metrics.frame_count == 3


def test_inference_latency_is_recorded():

    metrics = PerformanceMetrics()

    start = metrics.start_inference()

    time.sleep(0.01)

    elapsed = metrics.end_inference(start)

    assert elapsed > 0
    assert metrics.get_average_latency_ms() > 0


def test_fps_is_positive_after_processing():

    metrics = PerformanceMetrics()

    metrics.update()

    time.sleep(0.01)

    assert metrics.get_fps() > 0