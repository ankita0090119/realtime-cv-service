import cv2


class VideoReader:

    def __init__(self, source):
        self.source = source
        self.capture = cv2.VideoCapture(source)

        if not self.capture.isOpened():
            raise RuntimeError(
                f"Could not open video source: {source}"
            )

    def read(self):
        success, frame = self.capture.read()

        if not success:
            return None

        return frame

    def get_timestamp(self):
        """
        Return the timestamp of the most recently
        read frame in seconds.
        """
        return (
            self.capture.get(
                cv2.CAP_PROP_POS_MSEC
            ) / 1000.0
        )

    def release(self):
        self.capture.release()