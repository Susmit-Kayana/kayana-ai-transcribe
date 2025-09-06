# Use official Python 3.12 slim image
FROM python:3.12-slim

# Set work directory
WORKDIR /app

# Install git (required for pip git installs) and other tools
RUN apt-get update && \
    apt-get install -y git curl build-essential ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Copy project files
COPY main.py .
COPY requirements.txt .
COPY download_models.py .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download WhisperX models (CPU/MPS by default)
RUN python download_models.py

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
