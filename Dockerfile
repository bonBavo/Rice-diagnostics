FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_DIR=/app/model \
    UPLOAD_DIR=/app/data/uploads \
    DB_PATH=/app/data/history.db \
    HOST=0.0.0.0 \
    PORT=5000 \
    FLASK_DEBUG=0

WORKDIR /app

COPY requirements-docker.txt .
RUN python -m pip install --no-cache-dir --retries 20 --timeout 900 --upgrade pip \
    && python -m pip install --no-cache-dir --retries 20 --timeout 900 -r requirements-docker.txt

COPY app/ ./app/
COPY templates/ ./templates/
COPY static/ ./static/
COPY model/ ./model/
COPY class_labels.py run.py ./

RUN mkdir -p /app/data/uploads

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD python -c "import json, urllib.request; response = urllib.request.urlopen('http://127.0.0.1:5000/health', timeout=5); assert json.load(response)['status'] == 'ok'" || exit 1

CMD ["gunicorn", "--bind=0.0.0.0:5000", "--workers=1", "--threads=4", "--timeout=120", "--access-logfile=-", "--error-logfile=-", "app:create_app()"]
