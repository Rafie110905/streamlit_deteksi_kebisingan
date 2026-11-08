FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install FastAPI + Uvicorn
COPY admin_panel/requirements.txt ./admin_requirements.txt
RUN python -m pip install --no-cache-dir -r admin_requirements.txt

# Copy modul logging dan admin panel
COPY deteksi_logger.py ./
COPY admin_panel/ ./admin_panel/

EXPOSE 8000

CMD ["uvicorn", "admin_panel.main:app", "--host", "0.0.0.0", "--port", "8000"]
