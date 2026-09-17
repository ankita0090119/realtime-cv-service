import asyncio
import time
import cv2

from fastapi import APIRouter, Request, WebSocket
from fastapi.responses import StreamingResponse
from fastapi.websockets import WebSocketDisconnect


router = APIRouter()


@router.get("/health")
def health_check():

    return {
        "status": "healthy",
        "service": "realtime-cv-service"
    }


@router.get("/metrics")
def get_metrics(request: Request):

    cv_service = request.app.state.cv_service

    return cv_service.get_metrics()


@router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket
):

    await websocket.accept()

    try:

        while True:

            cv_service = (
                websocket
                .app
                .state
                .cv_service
            )

            metrics = (
                cv_service.get_metrics()
            )

            await websocket.send_json(
                metrics
            )

            await asyncio.sleep(0.5)

    except WebSocketDisconnect:

        print("WebSocket client disconnected")


def generate_video_frames(request: Request):

    cv_service = (
        request.app.state.cv_service
    )

    last_frame_id = -1

    while True:

        result = (
            cv_service.get_latest_frame()
        )

        if result is None:

            time.sleep(0.01)
            continue

        frame, frame_id = result

        # Don't send the same frame repeatedly.
        if frame_id == last_frame_id:

            time.sleep(0.01)
            continue

        last_frame_id = frame_id

        # Convert OpenCV frame to JPEG
        success, buffer = cv2.imencode(
            ".jpg",
            frame,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                80
            ]
        )

        if not success:
            continue

        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n"
            b"Content-Length: "
            + str(len(frame_bytes)).encode()
            + b"\r\n\r\n"
            + frame_bytes
            + b"\r\n"
        )

@router.get("/video")
def video_stream(request: Request):

    return StreamingResponse(
        generate_video_frames(request),
        media_type=(
            "multipart/x-mixed-replace; "
            "boundary=frame"
        )
    )