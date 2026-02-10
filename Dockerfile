# -------------------------------------------------------------------
# AgentOS — Multi-stage Docker build for Azure Container Apps
# -------------------------------------------------------------------

# Stage 1: Install dependencies
FROM python:3.12-slim AS builder

WORKDIR /build

# Install system deps for compilation (some packages have C extensions)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        libffi-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --pre --prefix=/install -r requirements.txt

# -------------------------------------------------------------------
# Stage 2: Runtime image
# -------------------------------------------------------------------
FROM python:3.12-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY src/ src/
COPY demo/ demo/
COPY data/ data/

# Expose the API port
EXPOSE 8000

# Health check for container orchestrators
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Default environment variables (overridden by Container Apps config)
ENV PYTHONUNBUFFERED=1
ENV AGENTOS_DEMO=1

# Start the dashboard API server
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
