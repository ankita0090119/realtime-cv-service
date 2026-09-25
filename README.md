# D5 Real-Time Computer Vision Service

A production-oriented real-time computer vision service built with **Python, YOLO11, ByteTrack, OpenCV, FastAPI, WebSocket, and Docker**.

The system processes video, performs object detection and multi-object tracking, and provides real-time analytics including **zone occupancy, entry/exit events, dwell time, FPS, inference latency, and frame-drop statistics**.

---

## Features

- Real-time video processing with OpenCV
- YOLO11 object detection
- ByteTrack multi-object tracking with persistent IDs
- Zone-based occupancy analytics
- Entry/exit tracking
- Dwell-time calculation
- Bounded frame buffer with drop-oldest behavior
- FPS and inference-latency monitoring
- Frame-drop/backpressure monitoring
- FastAPI REST API
- WebSocket live metrics
- MJPEG annotated video streaming
- Health and readiness endpoints
- Live browser dashboard
- Docker support
- Unit tests with pytest
- PyTorch, ONNX Runtime, and OpenVINO benchmarking

---

## Architecture

```text
                         Video Source
                              │
                              ▼
                        VideoReader
                              │
                              ▼
                       FrameProducer
                         (thread)
                              │
                              ▼
                  Bounded FrameBuffer
                  (drop-oldest queue)
                              │
                              ▼
                         CVService
                              │
                              ▼
                    YOLO11 + ByteTrack
                              │
              ┌───────────────┼────────────────┐
              │               │                │
              ▼               ▼                ▼
       Zone Occupancy    Dwell Time      Performance
       Entry / Exit      Tracking        Metrics
              │               │                │
              └───────────────┼────────────────┘
                              │
                              ▼
                         FastAPI API
                              │
             ┌────────────────┼────────────────┐
             │                │                │
             ▼                ▼                ▼
        GET /metrics       WS /ws         GET /video
             │                │                │
             └────────────────┼────────────────┘
                              ▼
                       Live Dashboard
```

## Project Structure

```text
realtime-cv-service/
│
├── app/
│   ├── config.py
│   ├── logging_config.py
│   ├── main.py
│   │
│   ├── analytics/
│   │   ├── dwell_time.py
│   │   ├── metrics.py
│   │   ├── occupancy.py
│   │   └── zone.py
│   │
│   ├── api/
│   │   ├── routes.py
│   │   └── server.py
│   │
│   ├── detection/
│   │   ├── detector.py
│   │   └── factory.py
│   │
│   ├── services/
│   │   └── cv_service.py
│   │
│   └── stream/
│       ├── frame_buffer.py
│       ├── pipeline.py
│       └── video_reader.py
│
├── dashboard/
│   └── index.html
│
├── benchmarks/
│   ├── benchmark_onnx.py
│   ├── benchmark_onnx_direct.py
│   ├── benchmark_openvino.py
│   ├── benchmark_pipeline.py
│   ├── benchmark_pytorch.py
│   ├── check_video.py
│   └── compare_detection_tracking.py
│
├── tests/
│   ├── test_dwell_time.py
│   ├── test_frame_buffer.py
│   ├── test_metrics.py
│   └── test_zone.py
│
├── docs/
│   └── dashboard.png
│
├── Dockerfile
├── requirements.txt
├── LICENSE
└── README.md
```

## Tech Stack

- Python
- OpenCV
- YOLO11
- ByteTrack
- FastAPI
- WebSocket
- MJPEG
- PyTorch
- ONNX Runtime
- OpenVINO
- Docker
- pytest

## API

| Endpoint | Description |
| --- | --- |
| `GET /health` | Service health check |
| `GET /ready` | Service readiness check |
| `GET /metrics` | Current analytics and performance metrics |
| `GET /video` | Annotated MJPEG video stream |
| `WS /ws` | Real-time metrics |

## Interactive API documentation

`http://127.0.0.1:8000/docs`

## Configuration

Create a `.env` file:

```env
VIDEO_SOURCE=data/videos/CAM1.mp4
MODEL_PATH=yolo11n.pt
INFERENCE_BACKEND=pytorch
BUFFER_SIZE=2

ZONE_NAME=Product Area
ZONE_X1=1200
ZONE_Y1=200
ZONE_X2=1630
ZONE_Y2=880
```

## Run Locally

**1. Clone the repository**

```bash
git clone https://github.com/ankita0090119/realtime-cv-service.git
cd realtime-cv-service
```

**2. Create a virtual environment**

```bash
python -m venv venv
```

Activate it:

```bash
# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Start the service**

```bash
python -m uvicorn app.api.server:app --host 127.0.0.1 --port 8000
```

Open the dashboard at <http://127.0.0.1:8000/dashboard/>.

## Screenshot

![D5Vision Dashboard](docs/dashboard.png)

## Docker

**Build**

```bash
docker build -t realtime-cv-service .
```

**Run**

```bash
docker run --rm -p 8000:8000 -v "${PWD}\data:/app/data:ro" realtime-cv-service
```

## Testing

Run the test suite:

```bash
python -m pytest
```

The tests cover:

- Zone logic
- Frame buffering
- Dwell-time tracking
- Performance metrics

## Benchmarking

Benchmark scripts are available in the `benchmarks/` directory. They include:

- PyTorch inference
- ONNX Runtime inference
- OpenVINO inference
- Detection vs. tracking
- End-to-end pipeline performance

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
