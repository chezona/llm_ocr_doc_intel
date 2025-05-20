import logging
import os
import time

from conductor.client.configuration.configuration import Configuration
from conductor.client.http.api_client import ApiClient
from conductor.client.automator.task_handler import TaskHandler
# from dotenv import load_dotenv # No longer needed, pydantic-settings handles .env in config.py

# Import workers
from app.conductor_workers import process_document_task, store_document_data_task, process_human_corrected_data_task
from app.config import settings # Import the settings object
import app.db_utils as db_utils # Import db_utils for initialization

# Load environment variables from .env file
# load_dotenv() # Handled by app.config.settings

# Configure logging
# LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper() # Use settings.log_level
logging.basicConfig(
    level=settings.log_level.upper(), # Use log_level from settings, ensure it's upper
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler() 
    ]
)
logger = logging.getLogger(__name__)

# CONDUCTOR_SERVER_URL = os.getenv("CONDUCTOR_SERVER_URL", "http://localhost:8080/api") # Use settings.conductor_base_url

def main():
    logger.info(f"Starting Application Worker. Connecting to Conductor at: {settings.conductor_base_url}")

    # Initialize database schema
    try:
        logger.info("Initializing database...")
        db_utils.initialize_database()
        logger.info("Database initialization successful.")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}", exc_info=True)
        # Depending on the severity, you might want to exit or prevent worker startup
        # For now, we'll log and continue, but this could be a critical failure.

    config = Configuration(server_api_url=str(settings.conductor_base_url)) # Use conductor_base_url from settings, cast to str
    
    # Create TaskHandler
    task_handler = TaskHandler(
        workers=[
            process_document_task, 
            store_document_data_task,
            process_human_corrected_data_task
        ],
        configuration=config,
    )

    logger.info("Application Worker configured with HITL support. Starting task polling...")
    try:
        task_handler.start_processes()
        logger.info("Task polling started successfully.")
        while True:
            time.sleep(60) 
            logger.debug("Worker is alive...")
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Stopping worker...")
    except Exception as e:
        logger.error(f"An unexpected error occurred in the main worker loop: {e}", exc_info=True)
    finally:
        logger.info("Shutting down TaskHandler...")
        task_handler.stop_processes()
        logger.info("Application Worker stopped.")

if __name__ == '__main__':
    main() 