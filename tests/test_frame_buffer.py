from app.stream.frame_buffer import FrameBuffer


def test_frame_is_added_and_retrieved():

    buffer = FrameBuffer(max_size=2)

    frame = "frame-1"

    buffer.put(frame)

    assert buffer.size() == 1
    assert buffer.get() == frame
    assert buffer.size() == 0


def test_oldest_frame_is_dropped_when_buffer_is_full():

    buffer = FrameBuffer(max_size=2)

    buffer.put("frame-1")
    buffer.put("frame-2")
    buffer.put("frame-3")

    assert buffer.size() == 2

    assert buffer.get() == "frame-2"
    assert buffer.get() == "frame-3"


def test_empty_buffer_returns_none():

    buffer = FrameBuffer(max_size=2)

    assert buffer.get() is None