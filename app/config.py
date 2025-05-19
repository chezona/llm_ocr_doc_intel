"""
Configuration settings for the LLM-aided OCR application.

This module centralizes key configuration parameters such as API endpoints for 
Ollama and Conductor, and model names, making them easily accessible and modifiable.
"""
# Standard library imports
import os

# Third-party imports
from dotenv import load_dotenv

# Load environment variables from .env file
# This allows for easy configuration without modifying the code directly.
# Ensure a .env file exists in the project root for local development.
load_dotenv()

# --- Ollama Configuration --- #
# Base URL for the Ollama service. Default assumes it's running as 'ollama' service in Docker.
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")

# Name of the Ollama model to be used for primary text processing (extraction, classification).
OLLAMA_EXTRACTOR_MODEL_NAME = os.getenv("OLLAMA_EXTRACTOR_MODEL_NAME", "mistral:7b-instruct-q4_K_M")

# Name of the Ollama model to be used for text enhancement (correction).
# Using the same model as the extractor for simplicity and availability.
OLLAMA_TEXT_ENHANCER_MODEL_NAME = os.getenv("OLLAMA_TEXT_ENHANCER_MODEL_NAME", "mistral:7b-instruct-q4_K_M")

# Name of the Ollama model to be used for initial document classification.
# Defaults to the extractor model if not specified, but can be set to a different (e.g., smaller/faster) model.
OLLAMA_CLASSIFIER_MODEL_NAME = os.getenv("OLLAMA_CLASSIFIER_MODEL_NAME", OLLAMA_EXTRACTOR_MODEL_NAME)

# Name of the Ollama vision model to be used if performing OCR with Ollama (e.g., in ocr_utils).
OLLAMA_VISION_MODEL_NAME = os.getenv("OLLAMA_VISION_MODEL_NAME", "moondream:latest")

# Timeout for requests made to the Ollama service, in seconds.
OLLAMA_REQUEST_TIMEOUT = int(os.getenv("OLLAMA_REQUEST_TIMEOUT", "600"))

# --- OCR Configuration --- #
# Default OCR model can be specified here if needed, e.g., for a different Ollama vision model or Tesseract specific settings.
# Currently, ocr_utils.py handles model selection logic internally (e.g., preferring Tesseract if Ollama vision model fails or is not specified for OCR).
# Example: OCR_DEFAULT_OLLAMA_VISION_MODEL = os.getenv("OCR_DEFAULT_OLLAMA_VISION_MODEL", "llava:latest")

# --- API Configuration (for this application's potential future API) --- #
# Base URL if this application were to expose its own API endpoints.
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8080/api")

# --- Conductor Configuration --- #
# Base URL for the Conductor workflow server API.
# Ensure this matches your Conductor server setup (e.g., http://conductor-server:8080/api when running in Docker).
# For local testing if Conductor is run via docker-compose and this app is outside Docker, one might use: "http://localhost:8088/api"
CONDUCTOR_BASE_URL = os.getenv("CONDUCTOR_BASE_URL", "http://conductor-server:8080/api")

# --- File Path Configuration (for Docker volumes) --- #
# Path within the Docker container where project files (e.g., PDFs for processing) are mounted and accessible.
INPUT_FILE_PATH_IN_CONTAINER = "/usr/src/app/mounted_project_files"

# --- Logging Configuration --- #
# Default log level for the application.
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# --- Example: More Complex Configuration Structure (Pydantic-based) --- #
# If configurations become numerous or require validation, using Pydantic models is a good practice:
# from pydantic_settings import BaseSettings
# class AppSettings(BaseSettings):
#     ollama_base_url: str = "http://ollama:11434"
#     ollama_model_name: str = "mistral:7b-instruct-q4_K_M"
#     ollama_request_timeout: int = 600
#     conductor_base_url: str = "http://conductor-server:8080/api"
#     log_level: str = "INFO"
#     # Add other settings with types and defaults
#
#     class Config:
#         env_file = ".env"
#         env_file_encoding = "utf-8"
#
# settings = AppSettings()
# To use in other modules: from app.config import settings; print(settings.ollama_base_url)

# --- Example: Tesseract Configuration (if needed) --- #
# If Tesseract requires specific environment variables like TESSDATA_PREFIX:
# TESSERACT_CMD_PATH = os.getenv("TESSERACT_CMD_PATH", "tesseract") # Path to tesseract executable
# TESSDATA_PREFIX_PATH = os.getenv("TESSDATA_PREFIX_PATH") # Path to tessdata directory
# if TESSDATA_PREFIX_PATH:
#     os.environ["TESSDATA_PREFIX"] = TESSDATA_PREFIX_PATH 