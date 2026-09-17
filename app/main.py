import cv2

from detection.detector import Detector
from stream.video_reader import VideoReader
from analytics.zone import Zone
from analytics.occupancy import OccupancyCounter


def main():
    # Video input
    video = VideoReader("data/videos/CAM1.mp4")

    # YOLO + ByteTrack
    detector = Detector()

    # Occupancy counter
    occupancy_counter = OccupancyCounter()

    # Define the monitoring zone
    zone = Zone(
        name="Product Area",
        x1=1200,
        y1=200,
        x2=1630,
        y2=880
    )

    while True:
        # Read frame
        frame = video.read()

        if frame is None:
            break

        # Run YOLO + ByteTrack
        results = detector.predict(frame)

        result = results[0]

        # Draw YOLO detections and tracking IDs
        annotated_frame = result.plot()

        # IDs of people currently inside the zone
        inside_ids = []

        # Check whether tracking IDs exist
        if result.boxes.id is not None:

            # Bounding boxes
            boxes = result.boxes.xyxy.cpu().numpy()

            # ByteTrack IDs
            track_ids = result.boxes.id.cpu().numpy().astype(int)

            # Object classes
            classes = result.boxes.cls.cpu().numpy().astype(int)

            # Process every tracked object
            for box, track_id, class_id in zip(
                boxes,
                track_ids,
                classes
            ):

                # COCO class 0 = person
                if class_id != 0:
                    continue

                # Bounding box coordinates
                x1, y1, x2, y2 = box

                # Calculate center of bounding box
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                # Check whether person's center is inside zone
                if zone.contains(center_x, center_y):
                    inside_ids.append(track_id)

        # Update occupancy
        occupancy_counter.update(inside_ids)

        # Current number of people inside zone
        occupancy = occupancy_counter.count()

        # Draw monitoring zone
        cv2.rectangle(
            annotated_frame,
            (zone.x1, zone.y1),
            (zone.x2, zone.y2),
            (255, 0, 0),
            2
        )

        # Draw zone name
        cv2.putText(
            annotated_frame,
            zone.name,
            (zone.x1, zone.y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2
        )

        # Draw occupancy
        cv2.putText(
            annotated_frame,
            f"Occupancy: {occupancy}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # Display frame
        cv2.imshow(
            "Real-Time Detection",
            annotated_frame
        )

        # Press q to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Cleanup
    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()