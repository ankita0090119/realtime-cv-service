import queue


class FrameBuffer:

    def __init__(self, max_size=2):
        self.queue = queue.Queue(maxsize=max_size)

        self.total_frames = 0
        self.dropped_frames = 0

    def put(self, frame):

        self.total_frames += 1

        try:
            self.queue.put_nowait(frame)

        except queue.Full:

            # Remove the oldest frame
            try:
                self.queue.get_nowait()
                self.dropped_frames += 1

            except queue.Empty:
                pass

            # Add the newest frame
            try:
                self.queue.put_nowait(frame)

            except queue.Full:
                self.dropped_frames += 1
                return False

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

    def total_count(self):
        return self.total_frames

    def drop_ratio(self):

        if self.total_frames == 0:
            return 0.0

        return self.dropped_frames / self.total_frames