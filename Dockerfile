# -----------------------------
# Base image
# -----------------------------
FROM python:3.10-slim

# -----------------------------
# Install system dependencies
# -----------------------------
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------
# Set working directory
# -----------------------------
WORKDIR /app

# -----------------------------
# Copy requirements and install
# -----------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# -----------------------------
# Copy project files
# -----------------------------
COPY . .

# -----------------------------
# Expose port for Gunicorn
# -----------------------------
EXPOSE 10000

# -----------------------------
# Run Flask API with Gunicorn (Render sets $PORT)
# -----------------------------
CMD ["sh", "-c", "gunicorn -w 2 -b 0.0.0.0:${PORT:-10000} api:app"]
