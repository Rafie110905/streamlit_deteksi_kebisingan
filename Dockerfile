FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install system dependencies for audio processing and container healthcheck
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsndfile1 libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

# Create non-root user for security hardening
RUN useradd -m -u 1000 appuser

# Copy application source, model training dataset, and assets
COPY app_deteksi_kebisingan.py fitur_audio_kebisingan.csv peringatan_jangan_berisik.mp3 ./
COPY .streamlit/ .streamlit/

# Set correct ownership for non-root user
RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["python", "-m", "streamlit", "run", "app_deteksi_kebisingan.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
