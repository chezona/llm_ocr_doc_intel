"""
This module defines Conductor workers for an OCR and LLM processing pipeline.

It includes workers for OCR, LLM-based data extraction, and database storage.
These workers are managed by a TaskHandler and poll a Conductor server for tasks.
"""
from conductor.client.http.models.task_result import TaskResult
from conductor.client.http.models.task_result_status import TaskResultStatus
from conductor.client.worker.worker_interface import WorkerInterface
from conductor.client.configuration.configuration import Configuration
from conductor.client.automator.task_handler import TaskHandler
import time
import logging
from pydantic import ValidationError
import asyncio # Added for LangGraphPipelineWorker

from .config import (CONDUCTOR_BASE_URL, 
                     OLLAMA_EXTRACTOR_MODEL_NAME, 
                     OLLAMA_TEXT_ENHANCER_MODEL_NAME, 
                     OLLAMA_CLASSIFIER_MODEL_NAME # Added classifier model
                    )
from .ocr_utils import ocr_image_url, ocr_image_path
from .llm_utils import process_ocr_text_with_llm
from .graph_orchestrator import run_graph_pipeline, DocumentGraphState
from .config_models import OCRConfig, LLMConfig
from .models import ExtractedData, PSEGData
from .db_utils import initialize_database, insert_pseg_data

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LangGraphPipelineWorker(WorkerInterface):
    """
    Conductor worker that orchestrates a document processing pipeline using LangGraph.

    Receives an image URL or local file path. It then invokes the LangGraph pipeline
    which handles OCR, optional text enhancement, and structured data extraction.
    The final payload suitable for database storage is returned as output.
    """
    def execute(self, task):
        """
        Executes the LangGraph pipeline task dispatched by Conductor.
        """
        logger.info(f"Executing LangGraph Pipeline task: {task.task_id}, input: {task.input_data}")
        
        image_path_or_url = task.input_data.get('image_path_or_url')
        source_type = task.input_data.get('source_type') # Should be 'path' or 'url'

        if not image_path_or_url or not source_type:
            error_msg = "Missing image_path_or_url or source_type in task input"
            logger.error(f"Task {task.task_id}: {error_msg}")
            return TaskResult(
                task_id=task.task_id,
                workflow_instance_id=task.workflow_instance_id,
                status=TaskResultStatus.FAILED_WITH_TERMINAL_ERROR,
                output_data={'pipeline_error': error_msg}
            )

        try:
            # Define default configurations for the graph components
            # These could also be passed in via task.input_data if more flexibility is needed
            default_ocr_config = OCRConfig() # Uses defaults from Pydantic model
            
            default_enhancer_llm_config = None # Temporarily disabling enhancer for testing
            logger.info(f"Task {task.task_id}: Text enhancement is DISABLED for this run.")
            
            default_extractor_llm_config = LLMConfig(
                model_name=OLLAMA_EXTRACTOR_MODEL_NAME, # Use imported config
                temperature=0.1
            )
            
            default_classifier_llm_config = LLMConfig(
                model_name=OLLAMA_CLASSIFIER_MODEL_NAME, # Use imported config from app.config
                temperature=0.3 # Classification might benefit from slightly higher temp for confidence
            )
            # If OLLAMA_CLASSIFIER_MODEL_NAME is not set or empty, this will use the default from LLMConfig or .env
            # If OLLAMA_CLASSIFIER_MODEL_NAME itself defaults to OLLAMA_EXTRACTOR_MODEL_NAME, that's fine too.

            logger.info(f"Task {task.task_id}: Running LangGraph pipeline for {source_type} - {image_path_or_url}")
            logger.info(f"Task {task.task_id}: Extractor Model: {default_extractor_llm_config.model_name}, Enhancer Model: {default_enhancer_llm_config if default_enhancer_llm_config else 'None'}, Classifier Model: {default_classifier_llm_config.model_name}")
            
            # Run the async graph pipeline using asyncio.run()
            final_state: DocumentGraphState = asyncio.run(run_graph_pipeline(
                image_path_or_url=image_path_or_url,
                source_type=source_type,
                ocr_config=default_ocr_config,
                enhancer_llm_config=default_enhancer_llm_config,
                extractor_llm_config=default_extractor_llm_config,
                classifier_llm_config=default_classifier_llm_config # Pass the classifier config
            ))

            if final_state.get("error_message"):
                logger.error(f"Task {task.task_id}: LangGraph pipeline completed with error: {final_state['error_message']}")
                # Decide if this is a FAILED or FAILED_WITH_TERMINAL_ERROR based on the error
                # For now, let's use FAILED to allow retries if applicable by Conductor.
                return TaskResult(
                    task_id=task.task_id,
                    workflow_instance_id=task.workflow_instance_id,
                    status=TaskResultStatus.FAILED, # Or FAILED_WITH_TERMINAL_ERROR
                    output_data={'pipeline_error': final_state['error_message'], 'final_graph_state': final_state}
                )

            db_payload = final_state.get("db_storage_payload")
            if not db_payload:
                error_msg = "LangGraph pipeline did not produce a db_storage_payload."
                logger.error(f"Task {task.task_id}: {error_msg}")
                return TaskResult(
                    task_id=task.task_id,
                    workflow_instance_id=task.workflow_instance_id,
                    status=TaskResultStatus.FAILED_WITH_TERMINAL_ERROR,
                    output_data={'pipeline_error': error_msg, 'final_graph_state': final_state}
                )

            logger.info(f"Task {task.task_id}: LangGraph pipeline successful. DB Payload: {db_payload.get('document_type')}")
            return TaskResult(
                task_id=task.task_id,
                workflow_instance_id=task.workflow_instance_id,
                status=TaskResultStatus.COMPLETED,
                output_data=db_payload # This should match what DatabaseStorageWorker expects
            )

        except Exception as e:
            logger.error(f"LangGraph Pipeline task {task.task_id} failed: {e}", exc_info=True)
            return TaskResult(
                task_id=task.task_id,
                workflow_instance_id=task.workflow_instance_id,
                status=TaskResultStatus.FAILED, # Allow retries by Conductor
                output_data={'pipeline_error': str(e)}
            )

class DatabaseStorageWorker(WorkerInterface):
    """
    Conductor worker responsible for storing extracted document data into a database.

    Receives document type, raw text, and structured information. It reconstructs Pydantic
    models from the input and inserts data into the PostgreSQL table for PSEG bills via `db_utils`.
    """
    def execute(self, task):
        """
        Executes the database storage task dispatched by Conductor.

        Retrieves document type, raw text, and structured data from task input.
        Validates and inserts data into the PSEG bills table if applicable.
        Returns success or error status to Conductor.
        """
        logger.info(f"Executing Database Storage task: {task.task_id}")
        task_input = task.input_data
        
        doc_type = task_input.get('document_type')
        raw_text = task_input.get('raw_text')
        structured_info_dict = task_input.get('structured_info')

        if not raw_text:
            logger.warning(f"Missing raw_text for task {task.task_id}. Cannot store without it.")
            return TaskResult(
                task_id=task.task_id,
                workflow_instance_id=task.workflow_instance_id,
                status=TaskResultStatus.FAILED_WITH_TERMINAL_ERROR,
                output_data={'db_storage_error': 'Missing raw_text'}
            )

        if doc_type == 'pseg_bill' and structured_info_dict:
            try:
                pseg_data_to_store = PSEGData(**structured_info_dict)
                inserted_id = insert_pseg_data(pseg_data_to_store, raw_text, doc_type)
                if inserted_id:
                    logger.info(f"Successfully stored PSEG bill data with DB ID: {inserted_id}")
                    return TaskResult(
                        task_id=task.task_id,
                        workflow_instance_id=task.workflow_instance_id,
                        status=TaskResultStatus.COMPLETED,
                        output_data={'db_storage_status': 'success', 'db_id': inserted_id, 'document_type_stored': doc_type}
                    )
                else:
                    logger.error(f"Failed to store PSEG bill data in DB for task {task.task_id}")
                    return TaskResult(
                        task_id=task.task_id,
                        workflow_instance_id=task.workflow_instance_id,
                        status=TaskResultStatus.FAILED,
                        output_data={'db_storage_error': 'Insert PSEG returned no ID'}
                    )
            except ValidationError as ve:
                logger.error(f"Validation error creating PSEGData for DB storage: {ve}. Data: {structured_info_dict}")
                return TaskResult(
                    task_id=task.task_id,
                    workflow_instance_id=task.workflow_instance_id,
                    status=TaskResultStatus.FAILED_WITH_TERMINAL_ERROR,
                    output_data={'db_storage_error': f'PSEGData validation failed: {ve}'}
                )
            except Exception as e:
                logger.error(f"Error storing PSEG bill data to DB: {e}", exc_info=True)
                return TaskResult(
                    task_id=task.task_id,
                    workflow_instance_id=task.workflow_instance_id,
                    status=TaskResultStatus.FAILED,
                    output_data={'db_storage_error': str(e)}
                )
        
        elif doc_type != 'pseg_bill' or not structured_info_dict:
            reason = f"Document type is '{doc_type}' or structured_info is missing."
            if doc_type == 'invoice':
                reason = "General invoice DB storage not implemented."
            logger.info(f"{reason} No specific DB storage action for PSEG. Task: {task.task_id}")
            return TaskResult(
                task_id=task.task_id,
                workflow_instance_id=task.workflow_instance_id,
                status=TaskResultStatus.COMPLETED,
                output_data={'db_storage_status': 'skipped', 'reason': reason, 'document_type_processed': doc_type}
            )

        else:
            logger.warning(f"Unhandled document type '{doc_type}' or state in DatabaseStorageWorker. Task: {task.task_id}")
            return TaskResult(
                task_id=task.task_id,
                workflow_instance_id=task.workflow_instance_id,
                status=TaskResultStatus.FAILED,
                output_data={'db_storage_error': f'Unhandled document type: {doc_type}'}
            )

def start_workers(conductor_api_url: str):
    """
    Initializes the database and starts all Conductor task workers for the pipeline.

    This function now registers the LangGraphPipelineWorker and the DatabaseStorageWorker.
    The OCRTaskWorker and LLMProcessingTaskWorker are no longer used with the LangGraph architecture.
    """
    # Initialize DB (create tables if they don't exist) *before* starting workers
    logger.info("Initializing database before starting workers...")
    initialize_database()
    logger.info("Database initialization complete.")

    configuration = Configuration(server_api_url=conductor_api_url)
    task_handler = TaskHandler(
        workers=[
            LangGraphPipelineWorker(task_definition_name='langgraph_pipeline_task'),
            DatabaseStorageWorker(task_definition_name='db_storage_task')
        ],
        configuration=configuration
    )
    logger.info(f"Starting Conductor workers, polling from: {conductor_api_url}")
    logger.info("Waiting 30 seconds before starting to poll for tasks...")
    time.sleep(30) # Added delay
    task_handler.start_processes() # Ensure this is start_processes()

    # Keep the main thread alive, or the worker processes will be daemonized and exit
    try:
        while True:
            time.sleep(60) # Keep alive, check status or add other logic if needed
            # You might want to add a way to gracefully shut down here
            # For example, by checking a flag or an event
    except KeyboardInterrupt:
        logger.info("Shutting down workers...")
        task_handler.stop_processes() # Ensure workers are stopped gracefully
        logger.info("Workers stopped.")

if __name__ == '__main__':
    """
    Main entry point for the worker script.
    
    Configures logging and calls `start_workers` to initialize the database
    and launch the Conductor task workers, connecting to the Conductor server URL
    defined in the application's configuration.
    """
    # Configure logging
    # Set up basic logging if not already configured by importing module
    logger.info(f"Starting workers, connecting to Conductor at: {CONDUCTOR_BASE_URL}")
    # Configure Conductor client
    # Note: conductor_api_url parameter in start_workers is effectively CONDUCTOR_BASE_URL from config
    start_workers(CONDUCTOR_BASE_URL) 