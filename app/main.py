import cv2

from detection.detector import Detector
from stream.video_reader import VideoReader
from analytics.zone import Zone
from analytics.occupancy import OccupancyCounter
from analytics.dwell_time import DwellTimeTracker


def main():
    # --------------------------------------------------
    # 1. Video source
    # --------------------------------------------------
    video = VideoReader("data/videos/CAM1.mp4")

    # --------------------------------------------------
    # 2. YOLO + ByteTrack
    # --------------------------------------------------
    detector = Detector()

    # --------------------------------------------------
    # 3. Analytics components
    # --------------------------------------------------
    occupancy_counter = OccupancyCounter()
    dwell_tracker = DwellTimeTracker()

    # --------------------------------------------------
    # 4. Define monitoring zone
    # --------------------------------------------------
    zone = Zone(
        name="Product Area",
        x1=1200,
        y1=200,
        x2=1630,
        y2=880
    )

    # --------------------------------------------------
    # 5. Process video frame by frame
    # --------------------------------------------------
    while True:

        # Read frame
        frame = video.read()

        if frame is None:
            break

        # Video timestamp in seconds
        timestamp = (
            video.capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        )

        # --------------------------------------------------
        # 6. YOLO detection + ByteTrack tracking
        # --------------------------------------------------
        results = detector.predict(frame)

        result = results[0]

        # Draw detections and tracking IDs
        annotated_frame = result.plot()

        # IDs of people currently inside the zone
        inside_ids = []

        # --------------------------------------------------
        # 7. Process tracked objects
        # --------------------------------------------------
        if result.boxes.id is not None:

            # Bounding boxes
            boxes = result.boxes.xyxy.cpu().numpy()

            # Tracking IDs
            track_ids = (
                result.boxes.id
                .cpu()
                .numpy()
                .astype(int)
            )

            # Object classes
            classes = (
                result.boxes.cls
                .cpu()
                .numpy()
                .astype(int)
            )

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

                # Calculate center point
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                # --------------------------------------------------
                # 8. Zone membership
                # --------------------------------------------------
                if zone.contains(center_x, center_y):

                    inside_ids.append(track_id)

                    # Current dwell time
                    current_dwell = (
                        dwell_tracker.get_current_dwell(
                            track_id,
                            timestamp
                        )
                    )

                    # Display dwell time near person
                    cv2.putText(
                        annotated_frame,
                        f"{current_dwell:.1f}s",
                        (center_x, center_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2
                    )

        # --------------------------------------------------
        # 9. Update occupancy
        # --------------------------------------------------
        occupancy_counter.update(inside_ids)

        occupancy = occupancy_counter.count()

        # --------------------------------------------------
        # 10. Update dwell-time tracker
        # --------------------------------------------------
        entered_ids, exited_ids = dwell_tracker.update(
            inside_ids,
            timestamp
        )

        # --------------------------------------------------
        # 11. Draw zone
        # --------------------------------------------------
        cv2.rectangle(
            annotated_frame,
            (zone.x1, zone.y1),
            (zone.x2, zone.y2),
            (255, 0, 0),
            2
        )

        # Zone name
        cv2.putText(
            annotated_frame,
            zone.name,
            (zone.x1, zone.y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2
        )

        # --------------------------------------------------
        # 12. Display analytics
        # --------------------------------------------------

        # Current occupancy
        cv2.putText(
            annotated_frame,
            f"Occupancy: {occupancy}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # Total entries
        cv2.putText(
            annotated_frame,
            f"Total Entries: {dwell_tracker.total_entries}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        # Average completed dwell time
        cv2.putText(
            annotated_frame,
            f"Avg Dwell: "
            f"{dwell_tracker.get_average_dwell():.1f}s",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        # --------------------------------------------------
        # 13. Display frame
        # --------------------------------------------------
        cv2.imshow(
            "Real-Time Detection",
            annotated_frame
        )

        # Press q to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # --------------------------------------------------
    # 14. Cleanup
    # --------------------------------------------------
    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()