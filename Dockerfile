FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsndfile1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

COPY app_deteksi_kebisingan.py fitur_audio_kebisingan.csv ./
COPY .streamlit/ .streamlit/

EXPOSE 8501

CMD ["python", "-m", "streamlit", "run", "app_deteksi_kebisingan.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
