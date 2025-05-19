"""
Configuration settings for the LLM-aided OCR application.

This module centralizes all critical configuration parameters for the application, 
including API endpoints for external services like Ollama and Conductor,
LLM model names, timeouts, and file paths.

The primary goal is to make the application easily configurable without needing to 
modify the core logic. Configurations are loaded from environment variables,
with sensible defaults provided for common development setups (e.g., Docker Compose).
This approach supports different environments (development, testing, production)
by simply adjusting environment variables or the .env file.
"""
# Standard library imports
import os

# Third-party imports
from dotenv import load_dotenv

# Load environment variables from .env file.
# This line is crucial for local development, allowing developers to override 
# default configurations or provide sensitive credentials without hardcoding them.
# It searches for a .env file in the project root (or parent directories).
load_dotenv()

# --- Ollama Configuration ---
# This section defines how the application interacts with the Ollama service,
# which is responsible for serving local Large Language Models.

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
"""
Base URL for the Ollama service. 
Default ('http://ollama:11434') assumes Ollama is running as a service named 'ollama' 
in a Docker network, accessible on its default port. This is a common setup when using Docker Compose.
"""

OLLAMA_EXTRACTOR_MODEL_NAME = os.getenv("OLLAMA_EXTRACTOR_MODEL_NAME", "mistral:7b-instruct-q4_K_M")
"""
Specifies the default Ollama model used for the primary data extraction task 
(e.g., extracting structured data from PSEG bills). 
The default 'mistral:7b-instruct-q4_K_M' is a capable instruction-following model.
This can be overridden to use other models available in the Ollama instance.
"""

OLLAMA_TEXT_ENHANCER_MODEL_NAME = os.getenv("OLLAMA_TEXT_ENHANCER_MODEL_NAME", "mistral:7b-instruct-q4_K_M")
"""
Defines the Ollama model for the optional text enhancement step, which aims to correct
OCR errors. Defaulting to the same model as the extractor ensures availability and 
leverages a powerful model for this correction task if enabled.
"""

OLLAMA_CLASSIFIER_MODEL_NAME = os.getenv("OLLAMA_CLASSIFIER_MODEL_NAME", OLLAMA_EXTRACTOR_MODEL_NAME)
"""
Model used for the initial document classification step (e.g., identifying a document 
as a 'pseg_bill', 'invoice', etc.). 
It defaults to the `OLLAMA_EXTRACTOR_MODEL_NAME` for convenience but can be set 
to a different, potentially smaller or faster, model if classification is a distinct 
and less complex task than detailed extraction.
"""

OLLAMA_VISION_MODEL_NAME = os.getenv("OLLAMA_VISION_MODEL_NAME", "moondream:latest")
"""
Specifies the Ollama vision model used for OCR tasks if an Ollama-based OCR approach 
(e.g., using a multimodal model) is chosen over Tesseract. 
'moondream:latest' is an example of a small, fast vision model.
"""

OLLAMA_REQUEST_TIMEOUT = int(os.getenv("OLLAMA_REQUEST_TIMEOUT", "600"))
"""
Timeout in seconds for HTTP requests made to the Ollama service. 
A relatively long default (600 seconds = 10 minutes) is chosen because LLM inference, 
especially with larger models or on CPU, can be time-consuming. This prevents premature 
timeouts during processing.
"""

# --- OCR Configuration ---
# This section could hold more specific OCR configurations if needed, beyond just model names.
# For instance, Tesseract-specific parameters or thresholds for image preprocessing.

# Currently, ocr_utils.py selects the OCR method (Tesseract by default, or Ollama vision).
# Example for a different default Ollama vision model for OCR if OLLAMA_VISION_MODEL_NAME was too general:
# OCR_DEFAULT_OLLAMA_VISION_MODEL = os.getenv("OCR_DEFAULT_OLLAMA_VISION_MODEL", "llava:latest")

# --- Conductor Configuration ---
# Defines how the application connects to the Netflix Conductor workflow orchestrator.

CONDUCTOR_BASE_URL = os.getenv("CONDUCTOR_BASE_URL", "http://conductor-server:8080/api")
"""
Base API URL for the Netflix Conductor server.
The default ('http://conductor-server:8080/api') assumes Conductor server is running 
as a service named 'conductor-server' in a Docker network. This is typical for 
Docker Compose deployments where services communicate via their service names.
"""

# --- File Path Configuration ---
# Manages paths used by the application, especially important when running in Docker
# and dealing with mounted volumes.

INPUT_FILE_PATH_IN_CONTAINER = "/usr/src/app/mounted_project_files"
"""
Defines the absolute path *inside the app_worker Docker container* where input files 
(e.g., PDFs to be processed) are expected to be found. This path corresponds to 
a volume mount defined in docker-compose.yml, which maps a directory from the 
host machine to this path in the container. This allows the application to access 
files from the host system.
"""

# --- Logging Configuration ---
# Basic logging setup. More sophisticated logging (e.g., to files, structured logging)
# would typically be configured in a dedicated logging setup module.

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
"""
Default log level for the application. Standard levels like INFO, DEBUG, WARNING, ERROR
are supported. This allows for adjusting verbosity of logs without code changes.
"""

# --- Advanced Configuration Example (using Pydantic) ---
# The commented-out section below demonstrates a more robust way to handle numerous
# configurations or configurations requiring validation, using Pydantic's BaseSettings.
# This is a recommended practice for larger applications to ensure type safety
# and provide clear, validated settings.
#
# from pydantic_settings import BaseSettings
#
# class AppSettings(BaseSettings):
#     ollama_base_url: str = "http://ollama:11434"
#     ollama_extractor_model_name: str = "mistral:7b-instruct-q4_K_M"
#     ollama_text_enhancer_model_name: str = "mistral:7b-instruct-q4_K_M"
#     ollama_classifier_model_name: str = OLLAMA_EXTRACTOR_MODEL_NAME # Or a specific default
#     ollama_vision_model_name: str = "moondream:latest"
#     ollama_request_timeout: int = 600
#     
#     conductor_base_url: str = "http://conductor-server:8080/api"
#     
#     input_file_path_in_container: str = "/usr/src/app/mounted_project_files"
#     log_level: str = "INFO"
#     
#     # Example of a Tesseract specific path (if needed)
#     # tesseract_cmd_path: str = "tesseract" 
#     # tessdata_prefix_path: Optional[str] = None # Using Optional from typing
#
#     class Config:
#         env_file = ".env" # Specifies the .env file to load
#         env_file_encoding = "utf-8"
#         extra = "ignore" # Ignore extra fields from environment
#
# settings = AppSettings()
#
# # To use in other modules:
# # from app.config import settings
# # print(settings.ollama_base_url)
# # if settings.tessdata_prefix_path:
# #     os.environ["TESSDATA_PREFIX"] = settings.tessdata_prefix_path

# Note: The API_BASE_URL variable seems unused in the current codebase and might be
# a remnant or for future use if the application itself were to expose an API.
# API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8080/api")

# --- Example: Tesseract Configuration (if needed) --- #
# If Tesseract requires specific environment variables like TESSDATA_PREFIX:
# TESSERACT_CMD_PATH = os.getenv("TESSERACT_CMD_PATH", "tesseract") # Path to tesseract executable
# TESSDATA_PREFIX_PATH = os.getenv("TESSDATA_PREFIX_PATH") # Path to tessdata directory
# if TESSDATA_PREFIX_PATH:
#     os.environ["TESSDATA_PREFIX"] = TESSDATA_PREFIX_PATH 