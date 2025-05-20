import instructor
import litellm
import logging
import os
from typing import Optional

# Removed DoclingDocument import as it's no longer used by the active extraction path
# from docling_core.datamodel.docling_document_datamodel import DoclingDocument
from app.llm_response_models import PSEGData
from app.config import settings

logger = logging.getLogger(__name__)

# Configuration for Ollama and LLM now comes from settings
# OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
# OLLAMA_MODEL_NAME = os.getenv("OLLAMA_EXTRACTOR_MODEL_NAME", "mistral:7b-instruct-q4_K_M")
# OLLAMA_REQUEST_TIMEOUT = int(os.getenv("OLLAMA_REQUEST_TIMEOUT", 300))
# LLM_CONTEXT_MAX_TOKENS = int(os.getenv("LLM_CONTEXT_MAX_TOKENS", 3000))

# Attempt to initialize tokenizer globally within the module for efficiency
_tokenizer = None
_tokenizer_name = None

try:
    from transformers import AutoTokenizer
    # Determine a suitable tokenizer based on the OLLAMA_MODEL_NAME from settings
    if "mistral" in settings.ollama_extractor_model_name.lower():
        _tokenizer_name = "mistralai/Mistral-7B-Instruct-v0.1"
    elif "llama" in settings.ollama_extractor_model_name.lower():
        _tokenizer_name = "hf-internal-testing/llama-tokenizer"
    else:
        _tokenizer_name = "gpt2"
        logger.warning(f"Unknown OLLAMA_MODEL_NAME '{settings.ollama_extractor_model_name}'. Using generic tokenizer '{_tokenizer_name}'. Accuracy may vary.")

    if _tokenizer_name:
        _tokenizer = AutoTokenizer.from_pretrained(_tokenizer_name)
        logger.info(f"Successfully loaded tokenizer: {_tokenizer_name}")

except ImportError:
    logger.warning("`transformers` library not found. Tokenizer will not be used. Context length will be managed by character count.")
except Exception as e:
    logger.warning(f"Could not load tokenizer '{_tokenizer_name}': {e}. Tokenizer will not be used. Context length will be managed by character count.")

# Patch litellm to work with instructor for Ollama, using settings
client = instructor.patch(litellm.LiteLLM(base_url=str(settings.ollama_base_url)), mode=instructor.Mode.JSON_SCHEMA)

# Removed _get_markdown_from_docling_doc function as it was specific to DoclingDocument input
# def _get_markdown_from_docling_doc(docling_doc: DoclingDocument) -> str:
#     ...

def extract_pseg_data_from_markdown(markdown_content: str, document_source_info: str = "Markdown content") -> Optional[PSEGData]:
    """
    Extracts structured PSEG data from Markdown text using an LLM.

    Args:
        markdown_content: The Markdown content string.
        document_source_info: A string indicating the source of the markdown (e.g., file path, originating parser type).

    Returns:
        A PSEGData object if extraction is successful, None otherwise.
    """
    if not markdown_content.strip():
        logger.warning(f"Cannot extract PSEG data: Markdown content from '{document_source_info}' is empty.")
        return None

    prompt = f"""
    Given the following Markdown text extracted from a PSEG utility bill, please extract the requested information.
    The text may be truncated if the original document was too long.
    Focus on extracting information relevant to a PSEG bill.
    Respond ONLY with the JSON object adhering to the PSEGData schema.

    Document Text (Markdown):
    ---BEGIN DOCUMENT TEXT---
    {markdown_content}
    ---END DOCUMENT TEXT---

    Extract the following fields: account_number, customer_name, service_address, billing_address, billing_date (which might be labeled as 'bill_date'), billing_period_start_date, billing_period_end_date, due_date, total_amount_due, previous_balance, payments_received, current_charges, and any line_items (each with description and amount).
    Ensure dates are in YYYY-MM-DD format.
    """

    try:
        context_length_info = f"{_tokenizer.encode(markdown_content, add_special_tokens=False).__len__()} tokens" if _tokenizer else f"{len(markdown_content)} characters"
        logger.info(f"Attempting PSEG data extraction from '{document_source_info}' with model: {settings.ollama_extractor_model_name} using Markdown context of approx. {context_length_info}.")
        
        response = client.chat.completions.create(
            model=settings.ollama_extractor_model_name,
            response_model=PSEGData,
            messages=[
                {"role": "system", "content": "You are an expert PSEG bill data extraction AI. Your output must be a valid JSON object matching the PSEGData schema. Dates should be YYYY-MM-DD. The input document text is in Markdown format."},
                {"role": "user", "content": prompt}
            ],
            max_retries=settings.llm_max_retries, 
            timeout=settings.ollama_request_timeout
        )
        logger.info(f"Successfully extracted PSEG data from '{document_source_info}'.")
        return response
    except Exception as e:
        logger.error(f"LLM extraction failed for PSEG data from '{document_source_info}': {e}", exc_info=True)
        return None

# Removed extract_pseg_data_with_docling_input function as it's no longer part of the active LlamaParse-only path
# def extract_pseg_data_with_docling_input(docling_doc: DoclingDocument, original_file_path: Optional[str] = None) -> Optional[PSEGData]:
#     ... 