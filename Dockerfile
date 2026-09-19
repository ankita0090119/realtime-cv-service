FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV YOLO_CONFIG_DIR=/tmp/ultralytics_config

RUN mkdir -p /tmp/ultralytics_config

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir \
    torch==2.14.0+cpu \
    torchvision==0.29.0+cpu \
    --index-url https://download.pytorch.org/whl/test/cpu

COPY app ./app
COPY dashboard ./dashboard
COPY yolo11n.pt ./yolo11n.pt

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.api.server:app", "--host", "0.0.0.0", "--port", "8000"]