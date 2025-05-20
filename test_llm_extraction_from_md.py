import instructor
import litellm
import logging
import argparse
from typing import Optional
import sys
import os

# Add app directory to sys.path to allow imports from app
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'app')))

from llm_response_models import PSEGData
from config import settings # Assuming your settings are in app/config.py

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Attempt to initialize tokenizer (copied from app/llm_extractor.py for context length logging)
_tokenizer = None
_tokenizer_name = None
try:
    from transformers import AutoTokenizer
    if "mistral" in settings.ollama_extractor_model_name.lower():
        _tokenizer_name = "mistralai/Mistral-7B-Instruct-v0.1"
    elif "llama" in settings.ollama_extractor_model_name.lower():
        _tokenizer_name = "hf-internal-testing/llama-tokenizer"
    else:
        _tokenizer_name = "gpt2"
    if _tokenizer_name:
        _tokenizer = AutoTokenizer.from_pretrained(_tokenizer_name)
except ImportError:
    logger.warning("`transformers` library not found for tokenizer. Context length will be by character count.")
except Exception as e:
    logger.warning(f"Could not load tokenizer '{_tokenizer_name}': {e}. Context length will be by character count.")


# Patch litellm to work with instructor for Ollama, using settings
# Ensure settings.ollama_base_url is a string
client = instructor.patch(litellm.LiteLLM(base_url=str(settings.ollama_base_url)), mode=instructor.Mode.JSON_SCHEMA)

def extract_pseg_data_from_md_content(markdown_content: str) -> Optional[PSEGData]:
    """
    Extracts structured data from Markdown content using an LLM.

    Args:
        markdown_content: The Markdown content string.

    Returns:
        A PSEGData object if extraction is successful, None otherwise.
    """
    if not markdown_content.strip():
        logger.warning("Cannot extract PSEG data: Markdown content is empty.")
        return None

    prompt = f"""
    Given the following text (extracted from a document, likely a PSEG utility bill), please extract the requested information.
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
        context_length_info = ""
        if _tokenizer:
            try:
                context_length_info = f"approx. {_tokenizer.encode(markdown_content, add_special_tokens=False).__len__()} tokens"
            except Exception as te:
                logger.warning(f"Error encoding markdown_content with tokenizer: {te}. Falling back to character count.")
                context_length_info = f"{len(markdown_content)} characters"
        else:
            context_length_info = f"{len(markdown_content)} characters"
            
        logger.info(f"Attempting PSEG data extraction with model: {settings.ollama_extractor_model_name} using context of {context_length_info}.")
        
        response = client.chat.completions.create(
            model=settings.ollama_extractor_model_name,
            response_model=PSEGData,
            messages=[
                {"role": "system", "content": "You are an expert PSEG bill data extraction AI. Your output must be a valid JSON object matching the PSEGData schema. Dates should be YYYY-MM-DD."},
                {"role": "user", "content": prompt}
            ],
            max_retries=settings.llm_max_retries, # Using setting for retries
            timeout=settings.ollama_request_timeout
        )
        logger.info("Successfully extracted PSEG data.")
        return response
    except Exception as e:
        logger.error(f"LLM extraction failed for PSEG data from Markdown: {e}", exc_info=True)
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract PSEG data from a Markdown file.")
    parser.add_argument("markdown_filepath", help="Path to the Markdown file in the PSEG folder.")
    
    args = parser.parse_args()

    # Construct the full path relative to the script's location if PSEG folder is at root
    script_dir = os.path.dirname(__file__)
    # Assuming PSEG folder is at the same level as the directory containing this script (if script is in root)
    # or one level up if script is in e.g. a 'scripts' folder.
    # For PSEG at root and script at root:
    # project_root = script_dir 
    # md_file_full_path = os.path.join(project_root, args.markdown_filepath) 
    # More robust: assume PSEG is relative to where script is run or use absolute path.
    # For simplicity, let's assume the user provides a path relative to the project root,
    # and the PSEG folder is at the project root.
    
    # If the path provided isn't absolute, assume it's relative to the project root.
    # This script itself is in the project root.
    if not os.path.isabs(args.markdown_filepath):
        md_file_full_path = os.path.join(os.getcwd(), args.markdown_filepath)
    else:
        md_file_full_path = args.markdown_filepath

    if not os.path.exists(md_file_full_path):
        logger.error(f"Markdown file not found: {md_file_full_path}")
        sys.exit(1)
        
    if not md_file_full_path.startswith(os.path.join(os.getcwd(), "PSEG")):
        logger.warning(f"The provided file path '{md_file_full_path}' does not seem to be inside the 'PSEG' directory at the project root. Please ensure the path is correct (e.g., PSEG/your_file.md).")
        # We can choose to exit or proceed. For now, let's proceed with a warning.

    logger.info(f"Reading Markdown file: {md_file_full_path}")
    
    try:
        with open(md_file_full_path, 'r', encoding='utf-8') as f:
            md_content = f.read()
    except Exception as e:
        logger.error(f"Error reading Markdown file {md_file_full_path}: {e}")
        sys.exit(1)

    extracted_data = extract_pseg_data_from_md_content(md_content)

    if extracted_data:
        logger.info("Extraction successful:")
        print(extracted_data.model_dump_json(indent=2))
    else:
        logger.error("Extraction failed or returned no data.") 