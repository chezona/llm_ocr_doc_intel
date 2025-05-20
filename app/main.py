import logging
import os
import time

from conductor.client.configuration.configuration import Configuration
# from conductor.client.http.api_client import ApiClient # Not directly used, Configuration handles it for TaskHandler
from conductor.client.automator.task_handler import TaskHandler
from conductor.client.worker.worker import Worker # New import
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

    # Configure Conductor client
    # The server_api_url should be just the base URL for the API endpoint (e.g., http://localhost:8080/api)
    # The configuration object handles appending /metadata/taskdefs etc.
    conductor_config = Configuration(server_api_url=str(settings.conductor_base_url)) 
    # conductor_config.debug = True # Optional: Enable for more verbose Conductor client logging

    # Define workers
    # Note: poll_interval and domain can be fine-tuned as needed.
    # task_definition_name must match the name in your Conductor workflow definition.
    workers = [
        Worker(
            task_definition_name="doc_processing_task",
            execute_function=process_document_task,
            poll_interval=0.1 # Poll every 100ms
        ),
        Worker(
            task_definition_name="store_document_data_task",
            execute_function=store_document_data_task,
            poll_interval=0.1
        ),
        Worker(
            task_definition_name="process_human_corrected_data_task",
            execute_function=process_human_corrected_data_task,
            poll_interval=0.1
        )
    ]
    
    # Create TaskHandler with the list of Worker objects
    task_handler = TaskHandler(
        workers=workers,
        configuration=conductor_config,
        # metrics_settings=None # Optional: for Prometheus metrics if needed
    )

    logger.info("Application Worker configured with HITL support. Starting task polling...")
    try:
        # Start polling for tasks in a blocking manner
        task_handler.start_processes() 
        logger.info("Task polling process initiated.")
        # Keep the main thread alive if start_processes is non-blocking or to add more logic
        while True:
            time.sleep(60) 
            logger.debug("Worker main thread alive...") # This log might be noisy, adjust as needed
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Stopping worker...")
    except Exception as e:
        logger.error(f"An unexpected error occurred in the main worker loop: {e}", exc_info=True)
    finally:
        logger.info("Shutting down TaskHandler...")
        if 'task_handler' in locals() and task_handler is not None:
            task_handler.stop_processes()
        logger.info("Application Worker stopped.")

if __name__ == '__main__':
    main() 