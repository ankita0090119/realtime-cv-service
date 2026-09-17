from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.services.cv_service import CVService


@asynccontextmanager
async def lifespan(app: FastAPI):

    print("Starting CV service...")

    # Create CV service
    cv_service = CVService()

    # Store service in FastAPI application state
    app.state.cv_service = cv_service

    # Start background CV processing
    cv_service.start()

    yield

    print("Stopping CV service...")

    # Graceful shutdown
    cv_service.stop()


app = FastAPI(
    title="Real-Time Computer Vision Service",
    version="1.0.0",
    lifespan=lifespan
)


# API routes
app.include_router(router)


# Serve dashboard
app.mount(
    "/dashboard",
    StaticFiles(directory="dashboard", html=True),
    name="dashboard"
)