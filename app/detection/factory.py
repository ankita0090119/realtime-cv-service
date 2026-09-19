from app.detection.detector import Detector


def create_detector(backend: str, model_path: str):
    backend = backend.lower()

    if backend == "pytorch":
        return Detector(model_path)

    raise ValueError(
        f"Unsupported inference backend: {backend}"
    )