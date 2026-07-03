FROM python:3.10-slim-bookworm

WORKDIR /app

# Install system dependencies required for image processing and PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install CPU-only PyTorch first to reduce container size and RAM usage
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 10000 7860 5000

# Dynamically bind to Render's PORT environment variable (defaulting to 10000 for Render / 7860 for HF Spaces)
CMD gunicorn --workers 2 --timeout 120 --bind 0.0.0.0:${PORT:-10000} app:app
