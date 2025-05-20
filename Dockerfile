# Use a base image with Python, matching your desired Python version
FROM python:3.11-slim

# Set working directory
WORKDIR /usr/src/app

# Install system dependencies
# - build-essential for some Python packages that might compile C extensions
# - curl for general utility
# poppler-utils was for pdf2image (docling path), likely not needed for LlamaParse only
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    # poppler-utils \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Poetry installation removed
# ENV POETRY_VERSION=1.8.2
# RUN pip install --no-cache-dir "poetry==${POETRY_VERSION}"

# Copy requirements.txt for dependency installation
COPY requirements.txt ./

# Install dependencies using pip and requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Poetry lock and install steps removed
# RUN poetry lock --no-update --no-interaction --no-ansi && \
#     poetry install --only main --no-interaction --no-ansi

# Copy the rest of the application code
# This should come after pip install to ensure dependencies are cached if only app code changes
COPY ./app ./app

# (Optional) Expose a port if your app serves HTTP, e.g., for health checks
# EXPOSE 8000

# Set the command to run the Conductor worker using module execution
CMD ["python", "-m", "app.main"] 