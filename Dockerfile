FROM python:3.13-alpine

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY main.py .

RUN adduser --disabled-password --gecos "" appuser

# Create start script that writes .env from Railway env vars and runs in SSE mode
RUN printf '#!/bin/sh\ncat > /app/.env << EOF\nTELEGRAM_API_ID=${TELEGRAM_API_ID}\nTELEGRAM_API_HASH=${TELEGRAM_API_HASH}\nTELEGRAM_SESSION_STRING=${TELEGRAM_SESSION_STRING}\nTELEGRAM_SESSION_NAME=${TELEGRAM_SESSION_NAME}\nWEBHOOK_URL=${WEBHOOK_URL}\nWEBHOOK_API_KEY=${WEBHOOK_API_KEY}\nAPI_ACCESS_TOKEN=${API_ACCESS_TOKEN}\nEOF\nexport MCP_TRANSPORT=sse\nexport PORT=${PORT:-8080}\nexec python /app/main.py\n' > /app/start.sh && \
    chmod +x /app/start.sh

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8080

CMD ["sh", "/app/start.sh"]
