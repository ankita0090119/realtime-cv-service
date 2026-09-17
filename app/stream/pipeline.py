import threading
import time

from stream.frame_buffer import FrameBuffer


class FrameProducer:

    def __init__(self, video_reader, buffer_size=2):

        self.video_reader = video_reader

        self.buffer = FrameBuffer(
            max_size=buffer_size
        )

        self.running = False
        self.thread = None

    def start(self):

        self.running = True

        self.thread = threading.Thread(
            target=self._capture_loop,
            daemon=True
        )

        self.thread.start()

    def _capture_loop(self):

        while self.running:

            # Read frame
            frame = self.video_reader.read()

            # End of video / camera failure
            if frame is None:

                self.running = False
                break

            # Get timestamp immediately after reading
            timestamp = (
                self.video_reader.get_timestamp()
            )

            # Store frame + metadata together
            packet = (
                frame,
                timestamp
            )

            self.buffer.put(packet)

            # Small sleep prevents unnecessary CPU usage
            time.sleep(0.001)

    def get_frame(self):

        return self.buffer.get()

    def stop(self):

        self.running = False

        if self.thread is not None:

            self.thread.join(
                timeout=1
            )

    def dropped_frames(self):

        return self.buffer.dropped_count()