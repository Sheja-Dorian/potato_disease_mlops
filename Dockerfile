FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=2 \
    TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 PORT=8000
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY app/ app/
COPY models/ models/
COPY data/train/ data/train/
COPY data/test/ data/test/

RUN chmod -R a+rwX /app
ENV HOME=/tmp
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD python -c "import urllib.request,os;urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\",8000)}/health')"
# Render injects $PORT; WEB_CONCURRENCY = uvicorn worker processes inside one container
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers ${WEB_CONCURRENCY:-1}
