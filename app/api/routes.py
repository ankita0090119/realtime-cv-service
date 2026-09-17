from fastapi import APIRouter, Request


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