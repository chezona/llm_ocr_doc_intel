"""
Orchestrates the document processing pipeline using LangGraph.

This module defines the state, nodes, and graph for a multi-step process:
1. Perform OCR on an input document (image or PDF).
2. Optionally enhance the OCR text using an LLM.
3. Extract structured data from the (potentially enhanced) text using another LLM call.
4. Prepare the data for database storage.
"""
import logging
from typing import TypedDict, Optional, Dict, Any, Literal
import asyncio # Required for running async node functions
import json
import os

from langgraph.graph import StateGraph, END

from app.config_models import OCRConfig, LLMConfig
from app.models import ExtractedData, PSEGData # PSEGData is used if extracted_data_model contains it
from app.ocr_utils import ocr_image_path, ocr_image_url # Synchronous, will need to be wrapped or called via asyncio.to_thread
from app.llm_text_enhancer import enhance_ocr_text # Async
from app.llm_utils import process_ocr_text_with_llm, classify_document_type # Added classify_document_type
from app.llm_response_models import SimpleClassificationOutput # Added for the new node

logger = logging.getLogger(__name__)

# --- State Definition ---
class DocumentGraphState(TypedDict):
    """
    Represents the state of the document processing graph.
    It carries data between different processing nodes.
    """
    # Initial inputs
    image_path_or_url: str
    source_type: Literal["path", "url"]
    ocr_config: Optional[OCRConfig] = None # For ocr_utils functions
    enhancer_llm_config: Optional[LLMConfig] = None # For llm_text_enhancer
    extractor_llm_config: LLMConfig # For llm_utils (data extraction)
    classifier_llm_config: Optional[LLMConfig] = None # Optional: if not provided, can default to extractor_llm_config model

    # Intermediate results
    raw_ocr_text: Optional[str] = None
    corrected_ocr_text: Optional[str] = None
    text_to_process: Optional[str] = None # General text after OCR/enhancement
    
    # Classification result
    classification_result: Optional[SimpleClassificationOutput] = None # Stores output from classification_node

    # Final extraction result (renamed from extracted_data_model)
    final_extracted_data_model: Optional[ExtractedData] = None
    
    # Output for database worker
    # This structure should match what DatabaseStorageWorker expects
    db_storage_payload: Optional[Dict[str, Any]] = None

    # Error handling and operational flags
    error_message: Optional[str] = None
    enhancement_skipped: bool = False
    current_step: Optional[str] = None # To track progress

# --- Node Functions ---

async def execute_ocr_node(state: DocumentGraphState) -> DocumentGraphState:
    """
    LangGraph node to perform OCR on the input document.
    """
    logger.info(f"Graph Node: Executing OCR for {state['image_path_or_url']}")
    state['current_step'] = "OCR"
    try:
        ocr_text_result = None
        # ocr_image_path and ocr_image_url are synchronous
        # For use in an async LangGraph, they should be run in a thread pool
        if state["source_type"] == "path":
            ocr_text_result = await asyncio.to_thread(
                ocr_image_path, 
                state["image_path_or_url"], 
                state.get("ocr_config"), 
                False # use_ollama for OCR is False by default in ocr_utils
            )
        elif state["source_type"] == "url":
            ocr_text_result = await asyncio.to_thread(
                ocr_image_url, 
                state["image_path_or_url"], 
                state.get("ocr_config"), 
                False # use_ollama for OCR is False by default in ocr_utils
            )
        else:
            state["error_message"] = f"Unsupported source_type: {state['source_type']}"
            logger.error(state["error_message"])
            return state

        if not ocr_text_result:
            state["error_message"] = "OCR process returned no text."
            logger.warning(state["error_message"])
            # Don't necessarily stop the graph; extraction might handle empty text, or it might be an error condition
        
        state["raw_ocr_text"] = ocr_text_result
        state["text_to_process"] = ocr_text_result # Default to raw, can be overwritten by enhancer
        logger.info(f"OCR Node: Text extracted (length: {len(ocr_text_result if ocr_text_result else '')}).")
        state["error_message"] = None # Clear previous errors if successful

    except FileNotFoundError as e:
        logger.error(f"OCR Node: File not found - {e}", exc_info=True)
        state["error_message"] = f"File not found: {state['image_path_or_url']}"
    except IOError as e:
        logger.error(f"OCR Node: IOError during processing - {e}", exc_info=True)
        state["error_message"] = f"IOError processing document: {e}"
    except Exception as e:
        logger.error(f"OCR Node: Unexpected error - {e}", exc_info=True)
        state["error_message"] = f"Unexpected error in OCR: {str(e)}"
    return state

async def enhance_text_node(state: DocumentGraphState) -> DocumentGraphState:
    """
    LangGraph node to enhance the OCR text using an LLM.
    This step is conditional and can be skipped.
    """
    logger.info("Graph Node: Text Enhancement")
    state['current_step'] = "TextEnhancement"
    raw_text = state.get("raw_ocr_text")
    enhancer_config = state.get("enhancer_llm_config")

    if state.get("error_message"): # Skip if OCR failed badly
        logger.warning("Skipping text enhancement due to previous error.")
        state["enhancement_skipped"] = True
        return state

    if not enhancer_config:
        logger.info("No LLM config provided for enhancer, skipping text enhancement.")
        state["enhancement_skipped"] = True
        return state
    
    if not raw_text or len(raw_text) < 50: # Arbitrary short text threshold
        logger.info("Raw OCR text is too short or empty, skipping enhancement.")
        state["enhancement_skipped"] = True
        return state

    try:
        logger.info(f"Enhancing text (length: {len(raw_text)}) with model: {enhancer_config.model_name}")
        # enhance_ocr_text is already async
        corrected_text = await enhance_ocr_text(raw_text, enhancer_config)
        state["corrected_ocr_text"] = corrected_text
        state["text_to_process"] = corrected_text # Use corrected text for next step
        state["enhancement_skipped"] = False
        logger.info(f"Text enhancement complete. Corrected text length: {len(corrected_text)}")
        state["error_message"] = None 
    except Exception as e:
        logger.error(f"Text Enhancement Node: Error - {e}", exc_info=True)
        state["error_message"] = f"Error during text enhancement: {str(e)}"
        state["enhancement_skipped"] = True # Mark as skipped due to error, fallback to raw_ocr_text for extraction
    return state

async def initial_classification_node(state: DocumentGraphState) -> DocumentGraphState:
    """
    LangGraph node to perform initial document classification.
    """
    logger.info("Graph Node: Initial Document Classification")
    state['current_step'] = "InitialClassification"
    text_to_classify = state.get("text_to_process")
    
    # Determine which LLM config to use for classification
    # Default to the extractor_llm_config's model if no specific classifier_llm_config is provided.
    # extractor_llm_config is guaranteed to be in the state as per DocumentGraphState type hint.
    classifier_model_name_to_use = state["extractor_llm_config"].model_name 

    classifier_config_from_state = state.get("classifier_llm_config")
    if classifier_config_from_state and classifier_config_from_state.model_name:
        classifier_model_name_to_use = classifier_config_from_state.model_name
    # If classifier_llm_config was provided but its model_name was empty/None, 
    # it will fall back to using the extractor_llm_config.model_name.

    if not text_to_classify:
        logger.warning("No text available for classification. Setting type to 'other'.")
        state["classification_result"] = SimpleClassificationOutput(document_type="other", confidence=0.0)
        state["error_message"] = state.get("error_message", "No text available for classification.")
        return state

    try:
        # classify_document_type is synchronous, run in thread
        classification_output: SimpleClassificationOutput = await asyncio.to_thread(
            classify_document_type,
            text_to_classify,
            classifier_model_name_to_use # Use the determined model name
        )
        state["classification_result"] = classification_output
        logger.info(f"Initial classification complete: {classification_output.document_type}, Confidence: {classification_output.confidence}")
        state["error_message"] = None 
    except Exception as e:
        logger.error(f"Initial Classification Node: Error - {e}", exc_info=True)
        state["error_message"] = f"Error during initial classification: {str(e)}"
        state["classification_result"] = SimpleClassificationOutput(document_type="other", confidence=0.0)
    return state

async def pseg_extraction_node(state: DocumentGraphState) -> DocumentGraphState:
    """
    LangGraph node for detailed data extraction specifically for PSEG bills.
    """
    logger.info("Graph Node: PSEG Bill Data Extraction")
    state['current_step'] = "PSEGExtraction"
    text_for_extraction = state.get("text_to_process")
    # Extractor config should be set in the initial state by the caller (LangGraphPipelineWorker)
    extractor_config = state["extractor_llm_config"]

    if not text_for_extraction:
        logger.warning("No text for PSEG extraction. Creating empty/error ExtractedData.")
        state["final_extracted_data_model"] = ExtractedData(
            raw_text=state.get("raw_ocr_text", ""), 
            document_type="pseg_bill", # Assumed pseg_bill as we are in this node
            summary="No text was provided for PSEG extraction.",
            structured_info={}
        )
        state["error_message"] = state.get("error_message", "No text for PSEG extraction.")
        return state

    try:
        logger.info(f"Extracting PSEG data from text (length: {len(text_for_extraction)}) using model: {extractor_config.model_name}")
        # process_ocr_text_with_llm is synchronous, it handles its own DocumentProcessor init
        extracted_data: ExtractedData = await asyncio.to_thread(
            process_ocr_text_with_llm, 
            text_for_extraction, 
            extractor_config.model_name # Pass the model name from the state's config
            # model_name for process_ocr_text_with_llm comes from OLLAMA_EXTRACTOR_MODEL_NAME in app.config
            # Its internal DocumentProcessor uses the PSEG-focused prompt.
        )
        # Ensure the document type from the detailed extraction matches, or log a warning.
        if extracted_data.document_type != "pseg_bill":
            logger.warning(f"PSEG Extraction node yielded document type '{extracted_data.document_type}' instead of 'pseg_bill'. Proceeding with this result.")
        
        state["final_extracted_data_model"] = extracted_data
        logger.info(f"PSEG extraction complete. Final DocType: {extracted_data.document_type}")
        state["error_message"] = None
    except Exception as e:
        logger.error(f"PSEG Extraction Node: Error - {e}", exc_info=True)
        state["error_message"] = f"Error during PSEG data extraction: {str(e)}"
        state["final_extracted_data_model"] = ExtractedData(
            raw_text=text_for_extraction, 
            document_type="pseg_bill", # Assumed
            summary=f"Error during PSEG data extraction: {str(e)}",
            structured_info={}
        )
    return state

async def prepare_other_output_node(state: DocumentGraphState) -> DocumentGraphState:
    """
    LangGraph node to prepare DB payload for 'other' or unhandled document types.
    """
    logger.info("Graph Node: Preparing DB Payload for 'Other' Type")
    state['current_step'] = "PrepareOtherOutput"
    
    doc_type = "other"
    confidence = 0.0
    if state.get("classification_result"):
        doc_type = state["classification_result"].document_type
        confidence = state["classification_result"].confidence or 0.0

    note = f"Document classified as '{doc_type}' (Confidence: {confidence:.2f}). No specialized extraction performed or applicable."
    if state.get("error_message"):
        note = f"Document processing resulted in classification '{doc_type}'. Error encountered: {state['error_message']}"

    state["final_extracted_data_model"] = ExtractedData(
        raw_text=state.get("text_to_process", state.get("raw_ocr_text", "")), 
        document_type=doc_type, 
        summary=note,
        structured_info={"note": note}
    )
    logger.info(f"Prepared 'other' type output: {doc_type}")
    # This node will also flow to prepare_db_payload_node which handles the db_storage_payload structure
    return state

async def prepare_db_payload_node(state: DocumentGraphState) -> DocumentGraphState:
    """
    LangGraph node to prepare the payload for the DatabaseStorageWorker.
    """
    logger.info("Graph Node: Preparing Final DB Payload")
    state['current_step'] = "PrepareDBPayload"
    extracted_data_model = state.get("final_extracted_data_model") # Use the new state field
    
    if not extracted_data_model:
        logger.warning("No final_extracted_data_model found to prepare DB payload.")
        error_summary = state.get("error_message", "No final extracted data model available from previous steps.")
        # Try to get classified type if available, otherwise default to 'error'
        final_doc_type = "error"
        if state.get("classification_result") and state["classification_result"].document_type:
            final_doc_type = state["classification_result"].document_type
        
        state["db_storage_payload"] = {
            "document_type": final_doc_type,
            "raw_text": state.get("text_to_process", state.get("raw_ocr_text", "")), 
            "structured_info": {"error": error_summary},
            "summary": error_summary
        }
        return state

    db_payload = {
        "document_type": extracted_data_model.document_type,
        "raw_text": extracted_data_model.raw_text,
        "structured_info": None,
        "summary": extracted_data_model.summary
    }

    # Handle PSEGData or other dict-based structured_info
    if extracted_data_model.document_type == "pseg_bill" and isinstance(extracted_data_model.structured_info, PSEGData):
        db_payload["structured_info"] = extracted_data_model.structured_info.model_dump()
    elif isinstance(extracted_data_model.structured_info, dict):
        db_payload["structured_info"] = extracted_data_model.structured_info
    else: # Fallback for unexpected types of structured_info
        db_payload["structured_info"] = {}
        
    state["db_storage_payload"] = db_payload
    logger.info(f"DB payload prepared for document type: {db_payload.get('document_type')}")
    return state

# --- Conditional Edges Logic ---
def route_based_on_classification(state: DocumentGraphState):
    """
    Determines the next step based on the classified document type.
    """
    classification_output = state.get("classification_result")
    if classification_output:
        doc_type = classification_output.document_type
        logger.info(f"Routing based on classification: {doc_type}")
        if doc_type == "pseg_bill":
            return "pseg_extractor"
        # Add routes for 'invoice', 'receipt' if/when those nodes are implemented
        # elif doc_type == "invoice":
        #     return "invoice_extractor" 
        else: # 'other', or unhandled types like 'invoice', 'receipt' for now
            return "prepare_other_output"
    logger.warning("No classification result found, routing to prepare_other_output by default.")
    return "prepare_other_output"

# --- Graph Definition ---
def build_document_processing_graph():
    """
    Builds and returns the LangGraph workflow for document processing.
    """
    workflow = StateGraph(DocumentGraphState)

    # Add nodes
    workflow.add_node("ocr", execute_ocr_node)
    workflow.add_node("enhancer", enhance_text_node)
    workflow.add_node("initial_classifier", initial_classification_node)
    workflow.add_node("pseg_extractor", pseg_extraction_node) 
    workflow.add_node("prepare_other_output", prepare_other_output_node)
    workflow.add_node("db_preparer", prepare_db_payload_node) # Final payload preparation for all paths

    # Define edges
    workflow.set_entry_point("ocr")
    workflow.add_edge("ocr", "enhancer")
    workflow.add_edge("enhancer", "initial_classifier")
    
    # Conditional routing after classification
    workflow.add_conditional_edges(
        "initial_classifier",
        route_based_on_classification,
        {
            "pseg_extractor": "pseg_extractor",
            "prepare_other_output": "prepare_other_output"
            # Add mappings for "invoice_extractor", etc. when implemented
        }
    )
    
    workflow.add_edge("pseg_extractor", "db_preparer")
    workflow.add_edge("prepare_other_output", "db_preparer") # 'Other' outputs also go to the final DB preparer
    
    workflow.add_edge("db_preparer", END)

    logger.info("Document processing graph compiled with 2-pass architecture.")
    return workflow.compile()

# --- Main Execution Function (called by Conductor worker) ---
async def run_graph_pipeline(
    image_path_or_url: str, 
    source_type: Literal["path", "url"],
    extractor_llm_config: LLMConfig, # For PSEG extractor node and default for classifier
    ocr_config: Optional[OCRConfig] = None,
    enhancer_llm_config: Optional[LLMConfig] = None,    
    classifier_llm_config: Optional[LLMConfig] = None # Specific config for classifier node
) -> DocumentGraphState:
    """
    Runs the entire document processing LangGraph pipeline.

    Args:
        image_path_or_url: Path or URL of the document image.
        source_type: 'path' or 'url'.
        extractor_llm_config: LLM configuration for the detailed PSEG extraction step.
        ocr_config: Configuration for the OCR process.
        enhancer_llm_config: LLM configuration for the text enhancement step (optional).
        classifier_llm_config: LLM configuration for the initial classification step (optional, will use extractor_llm_config's model if None).

    Returns:
        The final state of the graph after execution.
    """
    graph = build_document_processing_graph()
    
    initial_state = DocumentGraphState(
        image_path_or_url=image_path_or_url,
        source_type=source_type,
        ocr_config=ocr_config,
        enhancer_llm_config=enhancer_llm_config,
        extractor_llm_config=extractor_llm_config,
        classifier_llm_config=classifier_llm_config, # Pass it to the state
        # Initialize other fields to None or default values
        raw_ocr_text=None,
        corrected_ocr_text=None,
        text_to_process=None,
        classification_result=None,
        final_extracted_data_model=None,
        db_storage_payload=None,
        error_message=None,
        enhancement_skipped=False,
        current_step=None
    )
    
    logger.info(f"Running graph pipeline for: {image_path_or_url}")
    # Use .astream() for streaming events if needed, or .ainvoke() for final result
    final_state_result = await graph.ainvoke(initial_state)
    
    logger.info(f"Graph pipeline finished. Final step: {final_state_result.get('current_step')}, Error: {final_state_result.get('error_message')}")
    return final_state_result

# Example usage within this file (for direct testing if needed)
# Removing the __main__ block and main_graph_example function as per user request
# to focus on full system testing via Docker and Conductor. 