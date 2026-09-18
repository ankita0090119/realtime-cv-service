import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.services.cv_service import CVService
from app.logging_config import setup_logging


# Configure application logging
setup_logging()

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):

    logger.info("Starting CV service...")

    cv_service = CVService()

    app.state.cv_service = cv_service

    cv_service.start()

    logger.info("CV service started successfully")

    yield

    logger.info("Stopping CV service...")

    cv_service.stop()

    logger.info("CV service stopped")


app = FastAPI(
    title="Real-Time Computer Vision Service",
    version="1.0.0",
    lifespan=lifespan
)


app.include_router(router)


app.mount(
    "/dashboard",
    StaticFiles(
        directory="dashboard",
        html=True
    ),
    name="dashboard"
)