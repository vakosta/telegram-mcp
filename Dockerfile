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

CMD sh -c "python -c 'import main; print(\"import OK\")' 2>&1; echo '---'; env | grep TELEGRAM; echo '---'; mcp-proxy --host=0.0.0.0 --port=${PORT:-8080} -- python main.py"
