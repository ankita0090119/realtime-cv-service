import asyncio
import logging
import time

import cv2

from fastapi import (
    APIRouter,
    Request,
    WebSocket,
    HTTPException
)

from fastapi.responses import StreamingResponse
from fastapi.websockets import WebSocketDisconnect


router = APIRouter()

logger = logging.getLogger(__name__)


@router.get("/health")
def health_check():

    return {
        "status": "healthy",
        "service": "realtime-cv-service"
    }


@router.get("/ready")
def readiness_check(request: Request):

    cv_service = request.app.state.cv_service

    status = cv_service.get_status()

    if (
        status["processing"]
        and status["model_loaded"]
        and status["video_open"]
        and status["processing_error"] is None
    ):

        return {
            "status": "ready",
            **status
        }

    raise HTTPException(
        status_code=503,
        detail={
            "status": "not_ready",
            **status
        }
    )


@router.get("/metrics")
def get_metrics(request: Request):

    cv_service = request.app.state.cv_service

    return cv_service.get_metrics()


@router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket
):

    await websocket.accept()

    logger.info(
        "WebSocket client connected"
    )

    try:

        while True:

            cv_service = (
                websocket
                .app
                .state
                .cv_service
            )

            metrics = (
                cv_service
                .get_metrics()
            )

            await websocket.send_json(
                metrics
            )

            await asyncio.sleep(0.5)

    except WebSocketDisconnect:

        logger.info(
            "WebSocket client disconnected"
        )


def generate_video_frames(
    request: Request
):

    cv_service = (
        request.app
        .state
        .cv_service
    )

    last_frame_id = -1

    while True:

        result = (
            cv_service
            .get_latest_frame()
        )

        if result is None:

            time.sleep(0.01)

            continue

        frame, frame_id = result

        # Don't send the same frame repeatedly
        if frame_id == last_frame_id:

            time.sleep(0.01)

            continue

        last_frame_id = frame_id

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
            + str(
                len(frame_bytes)
            ).encode()
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