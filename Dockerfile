FROM python:3.13-alpine

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir mcp-proxy

COPY main.py .

RUN adduser --disabled-password --gecos "" appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8080

CMD sh -c "mcp-proxy --host=0.0.0.0 --port=${PORT:-8080} \
  -e TELEGRAM_API_ID=${TELEGRAM_API_ID} \
  -e TELEGRAM_API_HASH=${TELEGRAM_API_HASH} \
  -e TELEGRAM_SESSION_STRING=${TELEGRAM_SESSION_STRING} \
  -e TELEGRAM_SESSION_NAME=${TELEGRAM_SESSION_NAME} \
  -- python main.py"
