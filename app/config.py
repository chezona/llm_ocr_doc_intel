"""
Configuration settings for the LLM-aided OCR application.

This module centralizes all critical configuration parameters for the application,
including API endpoints for external services like Ollama and Conductor,
LLM model names, timeouts, and file paths, using Pydantic's BaseSettings.

Configurations are loaded from environment variables and .env files,
with sensible defaults provided. This approach supports different environments
(development, testing, production) by simply adjusting environment variables
or the .env file. Pydantic provides type checking and validation.
"""
# Standard library imports
import os
from typing import Optional # Keep Optional if we add optional fields later

# Third-party imports
# from dotenv import load_dotenv # No longer needed, pydantic-settings handles .env
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import HttpUrl # For URL validation

class AppSettings(BaseSettings):
    """
    Defines all application settings using Pydantic's BaseSettings.
    Values are loaded from environment variables (case-insensitive by default)
    and .env files.
    """
    # --- Ollama Configuration ---
    ollama_base_url: HttpUrl = "http://ollama:11434"
    """Base URL for the Ollama service."""

    ollama_extractor_model_name: str = "mistral:7b-instruct-q4_K_M"
    """Default Ollama model for primary data extraction."""

    ollama_request_timeout: int = 600
    """Timeout in seconds for HTTP requests to the Ollama service (default: 10 minutes)."""

    # --- Conductor Configuration ---
    conductor_base_url: HttpUrl = "http://conductor-server:8080/api"
    """Base API URL for the Netflix Conductor server."""

    # --- Database Configuration ---
    database_url: str = "postgresql://pseguser:psegpassword@postgres_db_service:5432/psegdb"
    """Full database connection string for PostgreSQL."""

    # --- File Path Configuration ---
    input_file_path_in_container: str = "/usr/src/app/mounted_project_files"
    """
    Absolute path *inside the app_worker Docker container* where input files
    (e.g., PDFs) are expected. Corresponds to a volume mount.
    """

    # --- Logging Configuration ---
    log_level: str = "INFO"
    """Default log level (INFO, DEBUG, WARNING, ERROR)."""
    
    # --- LLM Context Configuration from llm_extractor.py ---
    # Moved here for centralized configuration
    llm_context_max_tokens: int = 3000
    """Token limit for the context sent to LLM, leaving room for prompt and response schema."""

    model_config = SettingsConfigDict(
        env_file = ".env",          # Load .env file
        env_file_encoding = "utf-8", # Specify encoding
        extra = "ignore"            # Ignore extra fields from environment/env file
    )

# Instantiate the settings
settings = AppSettings()

# To use in other modules:
# from app.config import settings
# print(settings.ollama_base_url)

# The old way of loading individual os.getenv calls is now handled by AppSettings.
# The load_dotenv() call at the top is also handled by AppSettings.

