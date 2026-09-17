import cv2

from detection.detector import Detector
from stream.video_reader import VideoReader


def main():
    video = VideoReader("data/videos/CAM1.mp4")
    detector = Detector()

    while True:
        frame = video.read()

        if frame is None:
            break

        results = detector.predict(frame)

        annotated_frame = results[0].plot()

        cv2.imshow("Real-Time Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()