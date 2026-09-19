import cv2

VIDEO_PATH = "data/videos/CAM1.mp4"

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(
        f"Could not open video: {VIDEO_PATH}"
    )

width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

fps = cap.get(
    cv2.CAP_PROP_FPS
)

print("Video width:", width)
print("Video height:", height)
print("Video FPS:", fps)
print(
    "Resolution:",
    f"{width}x{height}"
)

cap.release()