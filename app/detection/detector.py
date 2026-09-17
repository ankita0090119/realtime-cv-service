from ultralytics import YOLO


class Detector:

    def __init__(self, model_path: str = "yolo11n.pt"):
        self.model = YOLO(model_path)

    def predict(self, frame):
        results = self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False
        )

        return results