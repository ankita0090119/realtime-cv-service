import os

from dotenv import load_dotenv


load_dotenv()


class Settings:

    def __init__(self):

        self.video_source = os.getenv(
            "VIDEO_SOURCE",
            "data/videos/CAM1.mp4"
        )

        self.model_path = os.getenv(
            "MODEL_PATH",
            "yolo11n.pt"
        )

        self.buffer_size = int(
            os.getenv(
                "BUFFER_SIZE",
                "2"
            )
        )

        self.zone_name = os.getenv(
            "ZONE_NAME",
            "Product Area"
        )

        self.zone_x1 = int(
            os.getenv(
                "ZONE_X1",
                "1200"
            )
        )

        self.zone_y1 = int(
            os.getenv(
                "ZONE_Y1",
                "200"
            )
        )

        self.zone_x2 = int(
            os.getenv(
                "ZONE_X2",
                "1630"
            )
        )

        self.zone_y2 = int(
            os.getenv(
                "ZONE_Y2",
                "880"
            )
        )


settings = Settings()