"""
This module provides advanced OCR capabilities for document processing.
It supports Ollama vision models (via HTTP API) for primary OCR and Tesseract
as a fallback, including PDF to image conversion and image preprocessing.
"""
import logging
import os
import tempfile
import subprocess
import asyncio
import re # Added for cleaning Ollama output
import base64 # For encoding images for Ollama API
import json # For parsing Ollama API responses
from typing import List, Optional, Union

import httpx # For making HTTP requests to Ollama API
import numpy as np
from PIL import Image
import cv2 # Requires opencv-python-headless
from pdf2image import convert_from_path
from pdf2image.exceptions import PDFInfoNotInstalledError, PDFPageCountError, PDFSyntaxError

from app.config_models import OCRConfig # Moved from added_flow
from app.config import OLLAMA_VISION_MODEL_NAME, OLLAMA_BASE_URL # Updated import for vision model

logger = logging.getLogger(__name__)

class AdvancedOCRProcessor:
    """
    Orchestrates OCR using Ollama vision models or Tesseract fallback.

    Handles PDF conversion, page selection, image preprocessing, and text extraction.
    Provides asynchronous methods for core processing and synchronous wrappers for external use.
    """

    def __init__(self, ocr_config: Optional[OCRConfig] = None):
        """
        Initializes the AdvancedOCRProcessor with optional OCR configuration.

        Sets up the OCR configuration and checks for the availability of
        Tesseract and the configured Ollama vision model.
        """
        self.config = ocr_config if ocr_config else OCRConfig()
        logger.info(f"Initialized AdvancedOCRProcessor with config: {self.config}")
        self.tesseract_available = False
        self.ollama_available = False
        self.ollama_ocr_model_available = False # e.g., llava
        # Run _check_dependencies_async in a blocking way during initialization.
        # This is acceptable here as it's a one-time setup cost.
        # For a fully async application, this might be handled differently (e.g., an async factory).
        asyncio.run(self._check_dependencies_async())

    async def _check_dependencies_async(self):
        """
        Asynchronously checks for Tesseract and Ollama availability and the OCR model.
        
        Verifies that Tesseract is installed and that the Ollama service is reachable
        and has the configured multimodal model (e.g., `moondream:latest`) available.
        Updates internal flags based on these checks.
        """
        try:
            process_tess = subprocess.run(["tesseract", "--version"], check=True, capture_output=True, text=True)
            self.tesseract_available = True
            logger.info(f"Tesseract OCR is available. Version: {process_tess.stdout.splitlines()[0]}")
        except (subprocess.SubprocessError, FileNotFoundError) as e:
            logger.warning(f"Tesseract OCR is not installed or not in PATH: {e}")
            self.tesseract_available = False

        try:
            async with httpx.AsyncClient(base_url=OLLAMA_BASE_URL, timeout=10.0) as client:
                response = await client.get("/api/tags")
                if response.status_code == 200:
                    self.ollama_available = True
                    logger.info(f"Ollama service is available at {OLLAMA_BASE_URL}.")
                    models_data = response.json()
                    available_models = [model.get("name") for model in models_data.get("models", [])]
                    if OLLAMA_VISION_MODEL_NAME in available_models:
                        self.ollama_ocr_model_available = True
                        logger.info(f"The configured Ollama vision model '{OLLAMA_VISION_MODEL_NAME}' is available via API.")
                    else:
                        logger.warning(f"The configured Ollama vision model '{OLLAMA_VISION_MODEL_NAME}' not found via Ollama API /api/tags. Available: {available_models}. Ollama OCR might not function as expected or may require pulling the model and ensuring it's loaded in the service.")
                else:
                    logger.warning(f"Ollama service responded with status {response.status_code} from {OLLAMA_BASE_URL}/api/tags. Ollama might not be fully operational.")
                    self.ollama_available = False
        except httpx.RequestError as e:
            logger.warning(f"Could not connect to Ollama service at {OLLAMA_BASE_URL}: {e}")
            self.ollama_available = False
            self.ollama_ocr_model_available = False

    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Preprocesses a PIL Image to potentially improve OCR accuracy.

        Converts the image to grayscale and applies Otsu's thresholding to binarize it.
        Returns the preprocessed image, or the original if an error occurs.
        """
        try:
            # Ensure image is in a format that OpenCV can handle (e.g., RGB before grayscale).
            # Some image modes (like 'P' with a palette, or 'RGBA') might not convert directly
            # to grayscale as expected by cv2.cvtColor without first converting to 'RGB'.
            img_array = np.array(image.convert('RGB')) 
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            _, binary_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            return Image.fromarray(binary_img)
        except Exception as e:
            logger.error(f"Error during image preprocessing: {e}", exc_info=True)
            return image # Return original on error

    def _convert_pdf_to_images(self, pdf_input: Union[str, bytes]) -> List[Image.Image]:
        """
        Converts a PDF document (from a file path or bytes) into a list of PIL Images.

        Respects `max_pages` and `skip_first_n_pages` from the OCRConfig to select specific pages.
        Raises RuntimeError if PDF conversion fails (e.g., Poppler utilities not found).
        """
        first_page = self.config.skip_first_n_pages + 1
        last_page = None
        if self.config.max_pages > 0:
            last_page = self.config.skip_first_n_pages + self.config.max_pages

        logger.info(f"Converting PDF to images. Pages: {first_page} to {last_page if last_page else 'end'}.")
        
        try:
            if isinstance(pdf_input, str):
                if not os.path.exists(pdf_input):
                    raise FileNotFoundError(f"PDF file not found: {pdf_input}")
                # Timeout for pdf2image conversion; can be adjusted if needed for very large/complex PDFs.
                images = convert_from_path(pdf_input, first_page=first_page, last_page=last_page, timeout=60)
            elif isinstance(pdf_input, bytes):
                # Timeout for pdf2image conversion; can be adjusted if needed for very large/complex PDFs.
                images = convert_from_path(None, pdf_bytes=pdf_input, first_page=first_page, last_page=last_page, timeout=60)
            else:
                raise TypeError("pdf_input must be a file path (str) or bytes.")
        except (PDFInfoNotInstalledError, PDFPageCountError, PDFSyntaxError) as e:
            logger.error(f"pdf2image error: {e}. Ensure Poppler utils are installed and in PATH.", exc_info=True)
            raise RuntimeError(f"PDF conversion failed due to Poppler issues: {e}") from e
        except Exception as e:
            logger.error(f"Failed to convert PDF to images: {e}", exc_info=True)
            raise RuntimeError(f"PDF conversion failed: {e}") from e
        
        logger.info(f"Converted {len(images)} pages from PDF.")
        return images

    async def _ocr_image_with_tesseract_async(self, image: Image.Image) -> str:
        """
        Asynchronously performs OCR on a single PIL Image using Tesseract OCR.

        Requires Tesseract to be installed and in the system PATH.
        Returns the extracted text, or an empty string if Tesseract is unavailable or fails.
        """
        if not self.tesseract_available:
            return ""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            image.save(tmp_file.name)
            tmp_file_path = tmp_file.name
        try:
            process = await asyncio.create_subprocess_exec(
                "tesseract", tmp_file_path, "stdout", "-l", "eng", # "-l eng" specifies English language for OCR
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            if process.returncode == 0:
                return stdout.decode().strip()
            else:
                logger.error(f"Tesseract OCR error: {stderr.decode()}")
                return ""
        finally:
            os.remove(tmp_file_path)

    async def _ocr_image_with_ollama_async(self, image: Image.Image, ollama_model: str = OLLAMA_VISION_MODEL_NAME, prompt: Optional[str] = None) -> str:
        """
        Asynchronously performs OCR on an image using an Ollama vision model via HTTP API.

        Requires Ollama service to be running and the specified model to be available.
        Returns extracted text, or an empty string if Ollama is unavailable or the call fails.
        """
        if not self.ollama_available or not self.ollama_ocr_model_available:
            logger.warning(f"Ollama not available or model '{ollama_model}' not found. Skipping Ollama OCR.")
            return ""

        with tempfile.NamedTemporaryFile(suffix=".png") as tmp_img_file:
            image.save(tmp_img_file.name, format="PNG")
            with open(tmp_img_file.name, "rb") as f:
                image_bytes = f.read()
        
        base64_image = base64.b64encode(image_bytes).decode("utf-8")

        actual_prompt = prompt if prompt else "Transcribe the text visible in the provided image. Output only the transcribed text."
        
        payload = {
            "model": ollama_model,
            "prompt": actual_prompt,
            "images": [base64_image],
            "stream": False  # Get the full response at once
        }

        try:
            async with httpx.AsyncClient(base_url=OLLAMA_BASE_URL, timeout=600.0) as client: # Increased timeout for potentially long OCR tasks
                logger.info(f"Sending OCR request to Ollama model '{ollama_model}' for an image.")
                # Increased timeout for the post request as well, as model inference can be slow.
                response = await client.post("/api/generate", json=payload, timeout=600.0) 
                response.raise_for_status() # Will raise an exception for 4XX/5XX responses
                
                response_data = response.json()
                extracted_text = response_data.get("response", "").strip()
                
                # Optional: Log full response for debugging if text is empty but no error
                if not extracted_text and response_data:
                     logger.warning(f"Ollama OCR returned empty text but successful response: {response_data}")
                
                logger.info(f"Ollama OCR successful. Text length: {len(extracted_text)}")
                return extracted_text

        except httpx.HTTPStatusError as e:
            error_content = e.response.text
            logger.error(f"Ollama API error (status {e.response.status_code}) for model '{ollama_model}': {error_content}", exc_info=False)
            # Check for specific model not found messages, though raise_for_status handles it
            if "model not found" in error_content.lower() or e.response.status_code == 404:
                 logger.error(f"It seems the model '{ollama_model}' is not available in the Ollama service. Please ensure it's pulled and loaded.")
            return ""
        except httpx.RequestError as e:
            logger.error(f"Request to Ollama API failed: {e}", exc_info=True)
            return ""
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON response from Ollama: {e}", exc_info=True)
            return ""
        except Exception as e: # Catch any other unexpected errors
            logger.error(f"An unexpected error occurred during Ollama OCR: {e}", exc_info=True)
            return ""

    async def _extract_text_from_single_image_async(self, image: Image.Image, use_ollama: bool) -> str:
        """
        Asynchronously extracts text from one preprocessed image, trying Ollama then Tesseract.

        Attempts OCR using the configured Ollama vision model if `use_ollama` is True,
        Ollama service is available, and the specified model is found.
        Falls back to Tesseract if Ollama is not selected, fails, or is unavailable.
        Returns extracted text, or an empty string if all methods fail.
        """
        if use_ollama and self.ollama_available and self.ollama_ocr_model_available:
            logger.info("Attempting OCR with Ollama vision model.")
            # Use the configured vision model by default for Ollama OCR attempt
            text = await self._ocr_image_with_ollama_async(image, ollama_model=OLLAMA_VISION_MODEL_NAME)
            if text: 
                return text
            logger.warning("Ollama OCR attempt failed or returned empty, falling back to Tesseract if available.")
        
        if self.tesseract_available:
            logger.info("Falling back to OCR with Tesseract.") # Updated log message
            return await self._ocr_image_with_tesseract_async(image)
        
        logger.error("No OCR method (Ollama or Tesseract) succeeded or is available for a page.")
        return ""

    def _run_ocr_on_images(self, images: List[Image.Image], use_ollama: bool) -> str:
        """
        Synchronously runs the OCR pipeline on a list of PIL Images.

        This method acts as a synchronous wrapper around the asynchronous OCR processing
        for individual images. It manages an asyncio event loop to execute the async
        `ocr_pipeline` coroutine.
        
        Args:
            images: A list of PIL.Image.Image objects to process.
            use_ollama: Boolean flag to indicate whether to attempt OCR with Ollama.

        Returns:
            A single string concatenating all extracted text from the images,
            with pages separated by a standard marker.
        """
        
        async def ocr_pipeline():
            page_texts = []
            for i, img in enumerate(images):
                logger.info(f"Preprocessing page {i + 1}/{len(images)}.")
                preprocessed_img = self._preprocess_image(img)
                logger.info(f"Extracting text from page {i + 1}/{len(images)}.")
                text = await self._extract_text_from_single_image_async(preprocessed_img, use_ollama)
                page_texts.append(text)
            return page_texts

        # asyncio.run() is used here to execute the async ocr_pipeline function
        # from this synchronous method. It creates a new event loop, runs the
        # coroutine until completion, and then closes the loop.
        all_page_texts = asyncio.run(ocr_pipeline())
        # Standardized page separator
        return "\n\n--- PAGE BREAK ---\n\n".join(filter(None, all_page_texts))

# --- Public Synchronous Functions --- (These will be the primary interface for app.worker)

def ocr_image_path(file_path: str, ocr_config: Optional[OCRConfig] = None, use_ollama: bool = False) -> str:
    """
    Performs OCR on a document (PDF or image) specified by a local file path.

    Initializes AdvancedOCRProcessor, converts PDF to images if necessary, preprocesses,
    and extracts text. It will attempt to use an Ollama vision model if `use_ollama` 
    is True and the service/model is available; otherwise, it falls back to Tesseract OCR.
    
    Args:
        file_path: Path to the local image or PDF file.
        ocr_config: Optional OCRConfiguration object.
        use_ollama: If True, attempt OCR with Ollama first. Defaults to False (uses Tesseract).

    Returns:
        A string containing all extracted text from the document.
        Returns an empty string if no text could be extracted or if the file cannot be processed.

    Raises:
        FileNotFoundError: If the specified `file_path` does not exist.
        IOError: If the image file cannot be opened or read, or PDF conversion fails.
    """
    logger.info(f"Starting OCR process for local file: {file_path} with use_ollama={use_ollama}")
    processor = AdvancedOCRProcessor(ocr_config)
    _, extension = os.path.splitext(file_path)
    images: List[Image.Image] = []

    if not os.path.exists(file_path):
        logger.error(f"File not found for OCR: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")

    if extension.lower() == '.pdf':
        images = processor._convert_pdf_to_images(file_path)
    else:
        try:
            img = Image.open(file_path)
            images.append(img)
        except Exception as e:
            logger.error(f"Failed to open image file {file_path}: {e}", exc_info=True)
            raise IOError(f"Could not open or read image file: {file_path}") from e
    
    if not images:
        logger.warning(f"No images found or extracted from {file_path}. Returning empty string.")
        return ""
    return processor._run_ocr_on_images(images, use_ollama)

def ocr_image_url(image_url: str, ocr_config: Optional[OCRConfig] = None, use_ollama: bool = False) -> str:
    """
    Performs OCR on an image or PDF document specified by a URL.

    Initializes AdvancedOCRProcessor, fetches the content from the URL, 
    converts PDF to images if necessary, preprocesses images, and extracts text.
    It will attempt to use an Ollama vision model if `use_ollama` is True and the 
    service/model is available; otherwise, it falls back to Tesseract OCR.

    Args:
        image_url: URL of the image or PDF file.
        ocr_config: Optional OCRConfiguration object.
        use_ollama: If True, attempt OCR with Ollama first. Defaults to False (uses Tesseract).

    Returns:
        A string containing all extracted text from the document.
        Returns an empty string if no text could be extracted or if the URL content cannot be processed.

    Raises:
        IOError: If the content cannot be fetched from the URL, is not a valid image/PDF,
                 or if PDF conversion fails.
    """
    logger.info(f"Starting OCR process for URL: {image_url} with use_ollama={use_ollama}")
    processor = AdvancedOCRProcessor(ocr_config)
    images: List[Image.Image] = []
    
    try:
        import requests # Keep import local to this function
        from io import BytesIO # Keep import local

        logger.info(f"Fetching content from URL: {image_url}")
        response = requests.get(image_url, timeout=30) # General timeout for request
        response.raise_for_status()
        content = response.content
        content_type = response.headers.get('content-type', '').lower()
        
        is_pdf = 'pdf' in content_type or image_url.lower().endswith('.pdf')
        if is_pdf:
            images = processor._convert_pdf_to_images(content) # Pass bytes directly
        else:
            img = Image.open(BytesIO(content))
            images.append(img)
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching URL {image_url}: {e}", exc_info=True)
        raise IOError(f"Failed to retrieve content from URL: {image_url}") from e
    except Exception as e:
        logger.error(f"An unexpected error occurred with URL {image_url}: {e}", exc_info=True)
        # Re-raise to signal failure to the caller
        raise IOError(f"Could not process content from URL: {image_url}") from e

    if not images:
        logger.warning(f"No images found or extracted from URL {image_url}. Returning empty string.")
        return ""
    return processor._run_ocr_on_images(images, use_ollama)

