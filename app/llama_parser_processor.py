# app/llama_parser_processor.py
import logging
from pathlib import Path
from typing import List, Optional

from llama_parse import LlamaParse
from llama_index.core.schema import Document as LlamaIndexDocument 
from app.config import settings

logger = logging.getLogger(__name__)

def parse_document_with_llamaparse(file_path: str | Path) -> Optional[str]:
    """
    Parses a document using LlamaParse and returns the combined Markdown content.

    Args:
        file_path: Path to the document file.

    Returns:
        A string containing the combined Markdown content if parsing is successful, None otherwise.
    """
    if not settings.llama_cloud_api_key:
        logger.error("LLAMA_CLOUD_API_KEY not configured. Cannot use LlamaParse.")
        return None

    try:
        parser = LlamaParse(
            api_key=str(settings.llama_cloud_api_key),  # Ensure API key is string
            result_type="markdown",
            verbose=settings.log_level.upper() == "DEBUG", # Set verbose based on general log level
            language="en"  # TODO: Potentially make this configurable via settings
        )
        
        logger.info(f"Starting document parsing with LlamaParse for: {file_path}")
        llama_documents: List[LlamaIndexDocument] = parser.load_data(str(file_path))
        
        if not llama_documents:
            logger.warning(f"LlamaParse returned no documents for: {file_path}")
            return None

        full_markdown_content = "\n\n".join([doc.text for doc in llama_documents if doc.text and doc.text.strip()])
        
        if not full_markdown_content.strip():
            logger.warning(f"LlamaParse produced empty Markdown content for: {file_path}")
            return None
            
        logger.info(f"Successfully parsed document with LlamaParse: {file_path}. Markdown length: {len(full_markdown_content)}")
        return full_markdown_content

    except Exception as e:
        logger.error(f"An unexpected exception occurred during LlamaParse processing for {file_path}: {e}", exc_info=True)
        return None 