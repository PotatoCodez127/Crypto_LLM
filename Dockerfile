# Use an official, clean Python 3.11 slim runtime as our foundation image
FROM python:3.11-slim

# Prevent Python from writing .pyc files and force unbuffered logging outputs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Establish the execution workspace directory inside the container
WORKDIR /app

# Install system utilities necessary for compiling dense math and quantitative wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements definitions first to take advantage of Docker layer caching
COPY requirements.txt .

# Install explicit dependency sheets cleanly inside the isolated environment
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code layers into the workspace
COPY . .

# Expose a default port placeholder for potential metric telemetry endpoints
EXPOSE 8080

# Default command launches the 24/7 continuous live market daemon loop
CMD ["python", "main_live.py"]