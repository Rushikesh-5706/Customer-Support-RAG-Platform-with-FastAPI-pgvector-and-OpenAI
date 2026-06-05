FROM python:3.11-slim

LABEL maintainer="support@nexora.io"
LABEL description="IntelliSupport Customer Support RAG Platform"
LABEL version="1.0.0"

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY intellisupport/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# Set working directory to the intellisupport package
WORKDIR /app

# Copy the entire intellisupport directory as the working content
COPY intellisupport/ .

# Copy .env for container runtime (docker-compose injects env vars)
COPY .env.example .env.example

# Do not run as root
RUN adduser --disabled-password --gecos "" appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health')" || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
