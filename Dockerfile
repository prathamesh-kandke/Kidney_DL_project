FROM python:3.12-slim

# Install system dependencies required to build pyarrow
RUN apt update -y && apt install -y --no-install-recommends \
    cmake \
    g++ \
    build-essential \
    awscli \
    && apt clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Default command
CMD ["python", "app.py"]