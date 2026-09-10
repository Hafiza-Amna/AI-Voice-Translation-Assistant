FROM python:3.10-slim

WORKDIR /app

# Install system dependencies (ffmpeg is required for audio processing)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# FastAPI will run on PORT environment variable, defaults to 8000
ENV PORT=8000
EXPOSE $PORT

# Start application
CMD uvicorn app.main:app --host 0.0.0.0 --port $PORT
