from ultralytics import YOLO


class Detector:
    def __init__(self, model_path: str = "yolo11n.pt"):
        self.model = YOLO(model_path)

    def predict(self, frame):
        results = self.model(frame, verbose=False)
        return results