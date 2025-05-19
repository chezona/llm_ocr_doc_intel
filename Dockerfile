# Use a base image with Python
FROM python:3.10-slim

# Set working directory
WORKDIR /usr/src/app

# Install system dependencies
# - Tesseract OCR and its English language pack
# - OpenCV dependencies (libgl1-mesa-glx can be important)
# - Other build tools if necessary for some Python packages
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    libtesseract-dev \
    poppler-utils \
    build-essential \
    postgresql-client \
    libgl1-mesa-glx \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file and install Python dependencies
COPY requirements.txt ./
# It's good practice to upgrade pip first
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY ./app ./app

# Make port 8000 available for an optional FastAPI health check or simple API
# This is not strictly for the worker but good practice if you add an API endpoint.
EXPOSE 8000

# Set the command to run the worker script
# This assumes your worker script is executable and starts the polling process
CMD ["python", "-m", "app.worker"] 