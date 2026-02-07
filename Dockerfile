# Use an official Python runtime as a parent image (Alpine-based for minimal vulnerabilities)
FROM python:3.13-alpine

# Set the working directory in the container
WORKDIR /app

# Prevent Python from writing pyc files to disc
ENV PYTHONDONTWRITEBYTECODE=1
# Ensure Python output is sent straight to terminal (useful for logs)
ENV PYTHONUNBUFFERED=1

# Install Node.js (needed for Supergateway) and npm
RUN apk add --no-cache nodejs npm

# Copy dependency definition files and install Python deps
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install Supergateway globally
RUN npm install -g supergateway

# Copy the rest of the application code
COPY main.py .

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos "" appuser && chown -R appuser:appuser /app
USER appuser

# Default port for Supergateway SSE (Railway overrides via PORT env var)
ENV PORT=8000
EXPOSE 8000

# Run telegram-mcp through Supergateway (stdio → SSE bridge)
CMD supergateway --stdio "python main.py" --port ${PORT}
