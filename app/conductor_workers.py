import logging
from typing import Dict, Any, Optional

from conductor.client.http.models.task import Task # New import
from conductor.client.http.models.task_result import TaskResult, TaskResultStatus # New imports

# Removed docling processor and related imports
# from app.docling_processor import parse_document_with_docling
from app.llama_parser_processor import parse_document_with_llamaparse
# Use only extract_pseg_data_from_markdown from llm_extractor
from app.llm_extractor import extract_pseg_data_from_markdown 
from app.llm_response_models import PSEGData
# Settings will be used to get LlamaParse API key implicitly by the processor
# from app.config import settings 
import app.db_utils as db_utils

# Removed DoclingDocument import as it's no longer used
# from docling_core.datamodel.docling_document_datamodel import DoclingDocument 

logger = logging.getLogger(__name__)

def process_document_task(task: Task) -> TaskResult:
    """Conductor worker task for document processing. Routes to HITL on LLM failure."""
    logger.info(f"Executing process_document_task for task_id: {task.task_id}")
    task_result = TaskResult(
        task_id=task.task_id,
        workflow_instance_id=task.workflow_instance_id,
        worker_id="process_document_worker"
    )

    try:
        input_data = task.input_data
        document_path = input_data.get('document_path')
        document_type_hint = input_data.get('document_type_hint', 'PSEG_BILL') 
        
        logger.info(f"Starting process_document_task for: {document_path}, type_hint: {document_type_hint}. Using LlamaParse.")
        
        if not document_path:
            logger.error("Document path not provided.")
            task_result.status = TaskResultStatus.FAILED
            task_result.reason_for_incompletion = "Document path not provided"
            task_result.add_log_entry('Document path not provided')
            return task_result

        logger.info(f"Using LlamaParse for document: {document_path}")
        markdown_content = parse_document_with_llamaparse(document_path)
        
        if not markdown_content:
            logger.error(f"LlamaParse processing failed or produced no content for: {document_path}")
            task_result.status = TaskResultStatus.FAILED
            task_result.reason_for_incompletion = "LlamaParse processing failed or no content"
            task_result.add_log_entry(f'LlamaParse failed or no content for {document_path}.')
            return task_result
        
        text_for_hitl_and_db = markdown_content

        pseg_data_extracted: Optional[PSEGData] = extract_pseg_data_from_markdown(
            markdown_content,
            document_source_info=f"LlamaParse ({document_path})"
        )

        if pseg_data_extracted:
            logger.info(f"LLM extraction successful for: {document_path} using LlamaParse.")
            db_storage_payload = {
                "pseg_data_dict": pseg_data_extracted.model_dump(),
                "full_ocr_text": text_for_hitl_and_db, 
                "document_type": document_type_hint
            }
            task_result.status = TaskResultStatus.COMPLETED
            task_result.add_output_data("status_doc_processing", "LLM_SUCCESS")
            task_result.add_output_data("db_storage_payload", db_storage_payload)
            task_result.add_log_entry(f"Successfully processed {document_path} with LlamaParse, LLM extraction successful.")
        else:
            logger.warning(f"LLM extraction failed for {document_path} using LlamaParse. Routing for human review.")
            human_review_input = {
                "full_ocr_text": text_for_hitl_and_db, 
                "document_type_hint": document_type_hint,
                "original_document_path": document_path 
            }
            task_result.status = TaskResultStatus.COMPLETED # Task itself completed, decision routes to HITL
            task_result.add_output_data("status_doc_processing", "LLM_FAILURE")
            task_result.add_output_data("human_review_input", human_review_input)
            task_result.add_log_entry(f"Processed {document_path} with LlamaParse, LLM extraction failed. Sent for human review.")
            
    except Exception as e:
        logger.error(f"Critical error in process_document_task: {str(e)}", exc_info=True)
        task_result.status = TaskResultStatus.FAILED
        task_result.reason_for_incompletion = f"Critical error: {str(e)}"
        task_result.add_output_data("status_doc_processing", "PIPELINE_ERROR") # Ensure this is set for decision node
        task_result.add_log_entry(f"Exception in process_document_task: {str(e)}")
    
    return task_result


# The following tasks (store_document_data_task, process_human_corrected_data_task) 
# primarily deal with the data after it has been processed (either by LLM or human).
# Their core logic of interacting with db_utils and handling PSEGData models 
# should remain largely the same, as they receive data structures that are independent 
# of whether Docling or LlamaParse was used upstream for the initial text/Markdown extraction.
# The key field `full_ocr_text` will now consistently contain Markdown if LlamaParse was used.

def store_document_data_task(task: Task) -> TaskResult:
    """Conductor worker task for storing successfully LLM-processed document data."""
    logger.info(f"Executing store_document_data_task for task_id: {task.task_id}")
    task_result = TaskResult(
        task_id=task.task_id,
        workflow_instance_id=task.workflow_instance_id,
        worker_id="store_document_data_worker" # It's good practice to have a worker_id
    )
    
    try:
        # Conductor task input is directly in task.input_data as a dictionary
        payload = task.input_data.get('db_storage_payload') 
        logger.info(f"Payload received: {payload is not None}")

        if not payload:
            logger.error("No db_storage_payload provided to store_document_data_task.")
            task_result.status = TaskResultStatus.FAILED
            task_result.reason_for_incompletion = "Missing db_storage_payload"
            task_result.add_log_entry('db_storage_payload not found in input.')
            return task_result

        pseg_data_dict = payload.get("pseg_data_dict")
        if not pseg_data_dict:
            logger.error("Missing pseg_data_dict in db_storage_payload.")
            task_result.status = TaskResultStatus.FAILED
            task_result.reason_for_incompletion = "Missing pseg_data_dict"
            task_result.add_log_entry('pseg_data_dict not found in payload.')
            return task_result

        try:
            pseg_data_model = PSEGData(**pseg_data_dict)
        except Exception as val_err: # Catch Pydantic validation error or other issues
            details = str(val_err)
            logger.error(f"Error reconstructing PSEGData model: {details}", exc_info=True)
            task_result.status = TaskResultStatus.FAILED
            task_result.reason_for_incompletion = f"PSEGData model reconstruction failed: {details}"
            task_result.add_log_entry(f"PSEGData validation/reconstruction error: {str(val_err)}")
            return task_result

        text_content_for_db = payload.get("full_ocr_text") 
        document_type = payload.get("document_type")

        if text_content_for_db is None or document_type is None:
            logger.error("Missing text_content_for_db or document_type in db_storage_payload.")
            task_result.status = TaskResultStatus.FAILED
            task_result.reason_for_incompletion = "Missing full_ocr_text or document_type"
            task_result.add_log_entry('full_ocr_text or document_type missing.')
            return task_result

        inserted_id = db_utils.insert_pseg_data(
            pseg_data=pseg_data_model, 
            full_ocr_text=text_content_for_db,
            doc_type=document_type
        )
        
        if inserted_id:
            logger.info(f"PSEG data stored successfully with ID: {inserted_id}")
            task_result.status = TaskResultStatus.COMPLETED
            task_result.add_output_data('db_insertion_id', str(inserted_id))
            task_result.add_log_entry(f"Data stored with ID: {inserted_id}")
        else:
            logger.error("Failed to store PSEG data, no ID returned from DB.")
            task_result.status = TaskResultStatus.FAILED
            task_result.reason_for_incompletion = "DB insertion failed to return ID"
            task_result.add_log_entry('DB insertion failed.')

    except Exception as e:
        logger.error(f"Critical error in store_document_data_task: {str(e)}", exc_info=True)
        task_result.status = TaskResultStatus.FAILED
        task_result.reason_for_incompletion = f"Critical error: {str(e)}"
        task_result.add_log_entry(f"Exception in store_document_data_task: {str(e)}")
    
    return task_result

def process_human_corrected_data_task(task: Task) -> TaskResult:
    """Conductor worker task for processing human-corrected data and storing it."""
    logger.info(f"Executing process_human_corrected_data_task for task_id: {task.task_id}")
    task_result = TaskResult(
        task_id=task.task_id,
        workflow_instance_id=task.workflow_instance_id,
        worker_id="process_human_corrected_data_worker"
    )

    try:
        input_data = task.input_data
        logger.info(f"Input keys: {list(input_data.keys()) if input_data else 'None'}")

        corrected_pseg_data_dict = input_data.get("corrected_pseg_data_dict")
        text_content_from_human_input = input_data.get("full_ocr_text") 
        document_type = input_data.get("document_type_hint")

        if not corrected_pseg_data_dict:
            error_msg = "Missing 'corrected_pseg_data_dict' in input for human corrected data."
            logger.error(error_msg)
            task_result.status = TaskResultStatus.FAILED
            task_result.reason_for_incompletion = error_msg
            task_result.add_log_entry(error_msg)
            return task_result
        
        if text_content_from_human_input is None or document_type is None:
            error_msg = f"Missing 'full_ocr_text' or 'document_type_hint' in input for human corrected data. OCR Text: {'present' if text_content_from_human_input else 'missing'}, Type: {'present' if document_type else 'missing'}"
            logger.error(error_msg)
            task_result.status = TaskResultStatus.FAILED
            task_result.reason_for_incompletion = error_msg
            task_result.add_log_entry(error_msg)
            return task_result

        try:
            pseg_data_model = PSEGData(**corrected_pseg_data_dict)
        except Exception as val_err:
            details = str(val_err)
            logger.error(f"Error reconstructing PSEGData model from human corrected data: {details}", exc_info=True)
            task_result.status = TaskResultStatus.FAILED
            task_result.reason_for_incompletion = f"Corrected PSEGData model reconstruction failed: {details}"
            task_result.add_log_entry(f"Corrected PSEGData validation error: {str(val_err)}")
            return task_result

        inserted_id = db_utils.insert_pseg_data(
            pseg_data=pseg_data_model,
            full_ocr_text=text_content_from_human_input,
            doc_type=document_type
        )

        if inserted_id:
            logger.info(f"Human corrected PSEG data stored successfully with ID: {inserted_id}")
            task_result.status = TaskResultStatus.COMPLETED
            task_result.add_output_data('db_insertion_id', str(inserted_id))
            task_result.add_output_data('status_human_review', 'PROCESSED_AND_STORED')
            task_result.add_log_entry(f"Human corrected data stored with ID: {inserted_id}")
        else:
            logger.error("Failed to store human corrected PSEG data, no ID returned from DB.")
            task_result.status = TaskResultStatus.FAILED
            task_result.reason_for_incompletion = "DB insertion of human corrected data failed"
            task_result.add_log_entry('DB insertion (human corrected) failed.')

    except Exception as e:
        logger.error(f"Critical error in process_human_corrected_data_task: {str(e)}", exc_info=True)
        task_result.status = TaskResultStatus.FAILED
        task_result.reason_for_incompletion = f"Critical error: {str(e)}"
        task_result.add_log_entry(f"Exception in process_human_corrected_data_task: {str(e)}")
    
    return task_result

# If you have other tasks defined, keep them below.
# Example:
# def another_task(task: Dict[str, Any]) -> Dict[str, Any]:
#     # ... logic ...
#     return {'status': 'COMPLETED', 'output': {}, 'logs': []} 