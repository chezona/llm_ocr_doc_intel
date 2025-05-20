import logging
# import os # Not used directly
from typing import Dict, Any, Optional

# Remove old Conductor client imports if they are not used by plain worker functions
# from conductor.client.http.models.task import Task 
# from conductor.client.http.models.task_result import TaskResult
# from conductor.client.http.models.task_result_status import TaskResultStatus
# from conductor.client.worker.worker_task import worker_task

from app.docling_processor import parse_document_with_docling
from app.llm_extractor import extract_pseg_data
from app.llm_response_models import PSEGData
# from app.config import settings # Settings are used by llm_extractor and db_utils directly
import app.db_utils as db_utils

logger = logging.getLogger(__name__)

# This worker will be registered with the Conductor client in main.py or similar

def process_document_task(task: Dict[str, Any]) -> Dict[str, Any]:
    '''Conductor worker task for document processing. Routes to HITL on LLM failure.'''
    try:
        input_data = task.get('inputData', {})
        document_path = input_data.get('document_path')
        # Default to PSEG_BILL if not provided, but allow override from workflow input
        document_type_hint = input_data.get('document_type_hint', 'PSEG_BILL') 
        
        logger.info(f"Starting process_document_task for: {document_path}, type_hint: {document_type_hint}")
        
        if not document_path:
            logger.error("Document path not provided.")
            # This is a terminal failure for this task, can't proceed.
            return {'status': 'FAILED', 'output': {'error': 'Document path not provided'}, 'logs': ['Document path not provided']}

        docling_document = parse_document_with_docling(document_path)
        if not docling_document or not docling_document.text:
            logger.error("Docling processing failed or returned no text.")
            # This is also a terminal failure for this task.
            return {'status': 'FAILED', 'output': {'error': 'Docling processing failed or no text'}, 'logs': ['Docling processing returned no text/document.']}
        logger.info(f"Docling processed document. Text length: {len(docling_document.text)}")

        pseg_data_extracted: Optional[PSEGData] = extract_pseg_data(docling_document)
        
        if pseg_data_extracted:
            logger.info(f"LLM extraction successful for: {document_path}")
            db_storage_payload = {
                "pseg_data_dict": pseg_data_extracted.model_dump(),
                "full_ocr_text": docling_document.text,
                "document_type": document_type_hint
            }
            return {
                'status': 'COMPLETED', 
                'output': {
                    "status_doc_processing": "LLM_SUCCESS",
                    "db_storage_payload": db_storage_payload
                },
                'logs': [f"Successfully processed {document_path}, LLM extraction successful."]
            }
        else:
            logger.warning(f"LLM extraction failed for {document_path}. Routing for human review.")
            human_review_input = {
                "full_ocr_text": docling_document.text,
                "document_type_hint": document_type_hint,
                "original_document_path": document_path # For context to the human reviewer
            }
            return {
                'status': 'COMPLETED', 
                'output': {
                    "status_doc_processing": "LLM_FAILURE",
                    "human_review_input": human_review_input
                },
                'logs': [f"Processed {document_path}, LLM extraction failed. Sent for human review."]
            }
    except Exception as e:
        logger.error(f"Critical error in process_document_task: {str(e)}", exc_info=True)
        # Return FAILED status for the task itself in case of unexpected errors
        return {'status': 'FAILED', 'output': {'error': str(e), 'status_doc_processing': 'PIPELINE_ERROR'}, 'logs': [f"Exception in process_document_task: {str(e)}"]}

def store_document_data_task(task: Dict[str, Any]) -> Dict[str, Any]:
    '''Conductor worker task for storing successfully LLM-processed document data.'''
    try:
        # This task now expects its direct input (db_storage_payload) from the workflow.
        payload = task.get('inputData', {}).get('db_storage_payload') 
        logger.info(f"Starting store_document_data_task. Payload received: {payload is not None}")

        if not payload:
            logger.error("No db_storage_payload received for storage.")
            return {'status': 'FAILED', 'output': {'error': 'No db_storage_payload found.'}, 'logs': ['No db_storage_payload found.']}

        pseg_data_dict = payload.get("pseg_data_dict")
        if not pseg_data_dict:
            logger.error("PSEGData (as dict 'pseg_data_dict') not in db_storage_payload.")
            return {'status': 'FAILED', 'output': {'error': 'pseg_data_dict not in payload'}, 'logs': ['pseg_data_dict missing from db_storage_payload.']}
        
        try:
            pseg_data_model = PSEGData(**pseg_data_dict)
        except Exception as val_err:
            logger.error(f"Error reconstructing PSEGData from dict: {str(val_err)}", exc_info=True)
            details = val_err.errors() if hasattr(val_err, 'errors') else str(val_err)
            return {'status': 'FAILED', 'output': {'error': 'PSEGData model reconstruction failed', 'details': details}, 'logs': [f"PSEGData validation/reconstruction error: {str(val_err)}"]}

        full_ocr_text = payload.get("full_ocr_text")
        document_type = payload.get("document_type")

        if full_ocr_text is None or document_type is None:
            logger.error("Missing full_ocr_text or document_type in db_storage_payload.")
            return {'status': 'FAILED', 'output': {'error': 'Missing full_ocr_text or document_type'}, 'logs': ['full_ocr_text or document_type missing.']}

        logger.info(f"Attempting to store LLM-processed data for account: {pseg_data_model.account_number}")
        
        inserted_id = db_utils.insert_pseg_data(
            pseg_data=pseg_data_model, 
            full_ocr_text=full_ocr_text, 
            doc_type=document_type
        )

        if inserted_id is not None:
            msg = f"Successfully stored LLM-processed document data. ID: {inserted_id}"
            logger.info(msg)
            return {'status': 'COMPLETED', 'output': {"database_record_id": inserted_id, "message": msg}, 'logs': [msg]}
        else:
            msg = "Failed to store LLM-processed document data in database."
            logger.error(msg)
            return {'status': 'FAILED', 'output': {"error": "DB insertion failed for LLM-processed data"}, 'logs': [msg]}

    except Exception as e:
        logger.error(f"Critical error in store_document_data_task: {str(e)}", exc_info=True)
        return {'status': 'FAILED', 'output': {'error': str(e)}, 'logs': [f"Exception: {str(e)}"]}

def process_human_corrected_data_task(task: Dict[str, Any]) -> Dict[str, Any]:
    '''Conductor worker task for processing human-corrected data and storing it.'''
    try:
        input_data = task.get('inputData', {})
        logger.info(f"Starting process_human_corrected_data_task. Input: {input_data.keys() if input_data else 'None'}")

        corrected_pseg_data_dict = input_data.get("corrected_pseg_data_dict")
        # These fields are mapped from the initial doc_processing_task output that went to the HUMAN task's input
        full_ocr_text = input_data.get("full_ocr_text") 
        document_type = input_data.get("document_type_hint") # Align with what human_review_input produced

        if not corrected_pseg_data_dict:
            error_msg = "Missing 'corrected_pseg_data_dict' from human review task output."
            logger.error(error_msg)
            return {'status': 'FAILED', 'output': {'error': error_msg}, 'logs': [error_msg]}
        
        if full_ocr_text is None or document_type is None:
            error_msg = f"Missing 'full_ocr_text' or 'document_type_hint' in input for human corrected data. OCR Text: {'present' if full_ocr_text else 'missing'}, Type: {'present' if document_type else 'missing'}"
            logger.error(error_msg)
            return {'status': 'FAILED', 'output': {'error': error_msg}, 'logs': [error_msg]}

        try:
            pseg_data_model = PSEGData(**corrected_pseg_data_dict)
            logger.info(f"Successfully reconstructed PSEGData model from human corrected input for type {document_type}.")
        except Exception as val_err:
            logger.error(f"Error reconstructing PSEGData model from human input: {str(val_err)}", exc_info=True)
            details = val_err.errors() if hasattr(val_err, 'errors') else str(val_err)
            return {'status': 'FAILED', 'output': {'error': 'PSEGData model reconstruction failed from human input', 'details': details}, 'logs': [f"PSEGData validation/reconstruction error: {str(val_err)}"]}

        logger.info(f"Attempting to store human-corrected data for account: {pseg_data_model.account_number}, type: {document_type}")
        inserted_id = db_utils.insert_pseg_data(
            pseg_data=pseg_data_model,
            full_ocr_text=full_ocr_text,
            doc_type=document_type
        )

        if inserted_id is not None:
            msg = f"Successfully stored human-corrected document data. ID: {inserted_id}"
            logger.info(msg)
            return {'status': 'COMPLETED', 'output': {"database_record_id": inserted_id, "message": msg}, 'logs': [msg]}
        else:
            msg = "Failed to store human-corrected document data in database."
            logger.error(msg)
            return {'status': 'FAILED', 'output': {"error": "Database insertion failed for human-corrected data"}, 'logs': [msg]}

    except Exception as e:
        logger.error(f"Critical error in process_human_corrected_data_task: {str(e)}", exc_info=True)
        return {'status': 'FAILED', 'output': {'error': str(e)}, 'logs': [f"Exception: {str(e)}"]}

# If you have other tasks defined, keep them below.
# Example:
# def another_task(task: Dict[str, Any]) -> Dict[str, Any]:
#     # ... logic ...
#     return {'status': 'COMPLETED', 'output': {}, 'logs': []} 