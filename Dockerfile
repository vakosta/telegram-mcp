FROM python:3.13-alpine

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir mcp-proxy

COPY main.py .

RUN adduser --disabled-password --gecos "" appuser

# Create wrapper that loads env from file
RUN printf '#!/usr/bin/env python3\nimport os\nwith open("/app/.env") as f:\n    for line in f:\n        line = line.strip()\n        if "=" in line:\n            k, v = line.split("=", 1)\n            os.environ[k] = v\nexec(open("/app/main.py").read())\n' > /app/run_wrapper.py

# Create start script
RUN printf '#!/bin/sh\ncat > /app/.env << EOF\nTELEGRAM_API_ID=${TELEGRAM_API_ID}\nTELEGRAM_API_HASH=${TELEGRAM_API_HASH}\nTELEGRAM_SESSION_STRING=${TELEGRAM_SESSION_STRING}\nTELEGRAM_SESSION_NAME=${TELEGRAM_SESSION_NAME}\nWEBHOOK_URL=${WEBHOOK_URL}\nWEBHOOK_API_KEY=${WEBHOOK_API_KEY}\nEOF\nexec mcp-proxy --host=0.0.0.0 --port=${PORT:-8080} -- python /app/run_wrapper.py\n' > /app/start.sh && \
    chmod +x /app/start.sh

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8080

CMD ["sh", "/app/start.sh"]