import instructor
import litellm
import logging
import os
from typing import Optional

from docling_core.datamodel.docling_document_datamodel import DoclingDocument
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

def _get_text_from_docling_doc_with_token_limit(docling_doc: DoclingDocument) -> str:
    """
    Constructs a single string from DoclingDocument text blocks, respecting a token limit from settings.
    """
    context_parts = []
    current_token_count = 0

    if not _tokenizer: # Fallback to character limit if tokenizer failed to load
        logger.warning("Tokenizer not available, falling back to character limit for context preparation.")
        char_limit = settings.llm_context_max_tokens * 4 
        full_text_char_based = "\n".join([block.text for page in docling_doc.pages for block in page.blocks if block.text])
        if len(full_text_char_based) > char_limit:
            logger.info(f"Character-based context length ({len(full_text_char_based)}) > {char_limit}. Truncating.")
            return full_text_char_based[:char_limit]
        return full_text_char_based

    # Token-based context assembly
    for page_idx, page in enumerate(docling_doc.pages):
        for block_idx, block in enumerate(page.blocks):
            if block.text and block.text.strip():
                block_token_ids = _tokenizer.encode(block.text, add_special_tokens=False)
                if current_token_count + len(block_token_ids) <= settings.llm_context_max_tokens:
                    context_parts.append(block.text)
                    current_token_count += len(block_token_ids)
                else:
                    logger.info(
                        f"Token limit ({settings.llm_context_max_tokens}) reached for LLM context. "
                        f"Stopped at page {page_idx + 1}, block {block_idx + 1}. "
                        f"Total tokens collected: {current_token_count}."
                    )
                    return "\n".join(context_parts)
    
    logger.info(f"Full document context prepared with approximately {current_token_count} tokens.")
    return "\n".join(context_parts)


def extract_pseg_data(docling_doc: DoclingDocument) -> Optional[PSEGData]:
    """
    Extracts structured data from a PSEG DoclingDocument using an LLM, configured via settings.

    Args:
        docling_doc: The DoclingDocument object from docling parsing.

    Returns:
        A PSEGData object if extraction is successful, None otherwise.
    """
    if not docling_doc.pages:
        logger.warning("Cannot extract PSEG data: DoclingDocument has no pages.")
        return None

    full_text_context = _get_text_from_docling_doc_with_token_limit(docling_doc)
    
    if not full_text_context.strip():
        logger.warning("Cannot extract PSEG data: No text content found in DoclingDocument blocks after context preparation.")
        return None

    prompt = f"""
    Given the following text extracted from a PSEG utility bill, please extract the requested information.
    The text may be truncated if the original document was too long.
    Focus on extracting information relevant to a PSEG bill.
    Respond ONLY with the JSON object adhering to the PSEGData schema.

    Document Text:
    ---BEGIN DOCUMENT TEXT---
    {full_text_context}
    ---END DOCUMENT TEXT---

    Extract the following fields: account_number, customer_name, service_address, billing_address, billing_date (which might be labeled as 'bill_date'), billing_period_start_date, billing_period_end_date, due_date, total_amount_due, previous_balance, payments_received, current_charges, and any line_items (each with description and amount).
    Ensure dates are in YYYY-MM-DD format.
    """

    try:
        logger.info(f"Attempting PSEG data extraction with model: {settings.ollama_extractor_model_name} using context of approx. {_tokenizer.encode(full_text_context, add_special_tokens=False).__len__() if _tokenizer else len(full_text_context)} {'tokens' if _tokenizer else 'characters'}.")
        response = client.chat.completions.create(
            model=settings.ollama_extractor_model_name,
            response_model=PSEGData,
            messages=[
                {"role": "system", "content": "You are an expert PSEG bill data extraction AI. Your output must be a valid JSON object matching the PSEGData schema. Dates should be YYYY-MM-DD."},
                {"role": "user", "content": prompt}
            ],
            max_retries=2, 
            timeout=settings.ollama_request_timeout
        )
        logger.info("Successfully extracted PSEG data.")
        return response
    except Exception as e:
        logger.error(f"LLM extraction failed for PSEG data: {e}", exc_info=True)
        return None 