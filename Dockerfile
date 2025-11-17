FROM python:3.10-slim


ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy API code
COPY inference_api.py /app/inference_api.py

# Expose FastAPI port
EXPOSE 8000

# HF model id can be overridden at runtime
ENV HF_MODEL_ID="BakshiSan/deberta-v3-anli-r2"

# Run the API server
CMD ["uvicorn", "inference_api:app", "--host", "0.0.0.0", "--port", "8000"]