# Use a base image with Python, matching pyproject.toml
FROM python:3.11-slim

# Set working directory
WORKDIR /usr/src/app

# Install system dependencies
# - poppler-utils for pdf2image
# - build-essential for some Python packages that might compile C extensions
# - curl for general utility
# - postgresql-client (optional, if direct DB operations from container are needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    build-essential \
    # postgresql-client \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
ENV POETRY_VERSION=1.8.2
RUN pip install --no-cache-dir "poetry==${POETRY_VERSION}"

# Copy only files necessary for dependency installation first to leverage Docker cache
COPY pyproject.toml poetry.lock* ./

# Install dependencies
# --no-dev: Do not install development dependencies
# --no-interaction: Do not ask any interactive questions
# --no-ansi: Disable ANSI output
RUN poetry lock --no-update --no-interaction --no-ansi && \
    poetry install --only main --no-interaction --no-ansi

# Copy the rest of the application code
# This should come after poetry install to ensure dependencies are cached if only app code changes
COPY ./app ./app

# (Optional) Expose a port if your app serves HTTP, e.g., for health checks
# EXPOSE 8000

# Set the command to run the Conductor worker
CMD ["python", "app/main.py"] 