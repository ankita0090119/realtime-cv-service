import queue


class FrameBuffer:

    def __init__(self, max_size=2):
        self.queue = queue.Queue(maxsize=max_size)
        self.dropped_frames = 0

    def put(self, frame):
        try:
            self.queue.put_nowait(frame)

        except queue.Full:
            # Remove the oldest frame
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass

            # Add the newest frame
            try:
                self.queue.put_nowait(frame)
            except queue.Full:
                self.dropped_frames += 1
                return False

            self.dropped_frames += 1

        return True

    def get(self):
        try:
            return self.queue.get_nowait()
        except queue.Empty:
            return None

    def size(self):
        return self.queue.qsize()

    def dropped_count(self):
        return self.dropped_frames