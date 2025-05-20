import logging
from pathlib import Path

from docling import DocumentConverter, ConversionResult
from docling.datamodel.conversion_options import PdfPipelineOptions, OcrStrategy
from docling_core.datamodel.docling_document_datamodel import DoclingDocument
# from docling_core.datamodel.error_datamodel import DoclingError # Not explicitly used yet

logger = logging.getLogger(__name__)

def parse_document_with_docling(file_path: str | Path) -> DoclingDocument | None:
    """
    Parses a document using Docling with VLM-focused options.

    Args:
        file_path: Path to the document file.

    Returns:
        A DoclingDocument if parsing is successful, None otherwise.
    """
    converter = DocumentConverter()
    
    # Configure options for VLM-based parsing.
    # This assumes that the [vlm] extra has installed appropriate backends
    # and models (like SmolDocling) and docling will pick them up.
    # We prioritize VLM for OCR and layout.
    options = PdfPipelineOptions(
        ocr_strategy=OcrStrategy.AUTO,  # Let VLM handle OCR if possible
        # Specific VLM model name might be configurable here or via docling's default
        # if multiple VLM backends are available. For now, rely on defaults.
        # layout_model_name="smoldocling", # Example if direct specification is needed/supported
        # ocr_model_name="smoldocling",    # Example
    )

    try:
        logger.info(f"Starting document conversion for: {file_path}")
        result: ConversionResult = converter.convert(file_path=str(file_path), options=options)
        
        if result.document:
            logger.info(f"Successfully converted document: {file_path}")
            return result.document
        elif result.errors:
            logger.error(f"Errors encountered during docling conversion for {file_path}:")
            for error in result.errors:
                logger.error(f"- Code: {error.code}, Message: {error.message}, Details: {error.details}")
            return None
        else:
            logger.error(f"Docling conversion failed for {file_path} with no specific errors reported and no document produced.")
            return None

    except Exception as e:
        logger.error(f"An unexpected exception occurred during docling processing for {file_path}: {e}", exc_info=True)
        return None 