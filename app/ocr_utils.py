"""
This module provides Optical Character Recognition (OCR) capabilities for the document processing pipeline.
It's designed to extract text from various input sources like PDF files and images (from local paths or URLs).

Core Functionalities:
- **PDF Processing:** Converts PDF documents into a series of images for OCR.
  Handles page selection (max pages, skip pages) based on configuration.
- **Image Preprocessing:** Includes steps like converting images to grayscale and 
  applying binarization (Otsu's thresholding) to potentially improve OCR accuracy.
- **Dual OCR Engine Support:** 
    - **Ollama Vision Models:** Can utilize Ollama-served vision models (e.g., LLaVA, Moondream) 
      for OCR via their HTTP API. This allows leveraging powerful multimodal models.
    - **Tesseract OCR:** Uses Tesseract as a robust, well-established open-source OCR engine. 
      It often serves as a fallback if Ollama-based OCR is unavailable or fails.
- **Configuration-Driven:** OCR behavior is controlled by an `OCRConfig` object, allowing 
  for flexible settings for different use cases.
- **Asynchronous Operations:** Core OCR operations are implemented asynchronously (`async`/
  `await`) to allow for non-blocking I/O, especially useful when dealing with 
  external processes (Tesseract) or network requests (Ollama API).
- **Error Handling & Fallbacks:** Includes error handling for common issues like missing 
  dependencies (Poppler for PDFs, Tesseract), API errors, and provides fallbacks 
  (e.g., Tesseract if Ollama fails).

The primary class, `AdvancedOCRProcessor`, orchestrates these steps. Standalone 
functions are also provided for more direct OCR on image paths or URLs.

The choice of providing both Ollama vision and Tesseract allows for flexibility: 
Ollama models might offer better accuracy or contextual understanding for certain document 
types, while Tesseract is a reliable and widely available baseline.
"""
import logging
import os
import tempfile
import subprocess
import asyncio
import re # Added for cleaning Ollama output
import base64 # For encoding images for Ollama API
import json # For parsing Ollama API responses
from typing import List, Optional, Union, Tuple

import httpx # For making HTTP requests to Ollama API
import numpy as np
from PIL import Image, UnidentifiedImageError
import cv2 # Requires opencv-python-headless
from pdf2image import convert_from_path
from pdf2image.exceptions import PDFInfoNotInstalledError, PDFPageCountError, PDFSyntaxError

from app.config_models import OCRConfig # Moved from added_flow
from app.config import OLLAMA_VISION_MODEL_NAME, OLLAMA_BASE_URL, LOG_LEVEL # Updated import for vision model, added LOG_LEVEL

logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL) # Set logger level from config

class AdvancedOCRProcessor:
    """
    Orchestrates the OCR process, handling PDF to image conversion, image preprocessing,
    and text extraction using either Ollama vision models or Tesseract OCR as a fallback.

    This class is designed to be the main entry point for performing OCR on documents.
    It takes an `OCRConfig` object to customize its behavior. The initialization includes
    checking for necessary dependencies like Tesseract and the availability of the 
    configured Ollama service and vision model. This proactive check helps in identifying 
    setup issues early.
    """

    def __init__(self, ocr_config: Optional[OCRConfig] = None):
        """
        Initializes the AdvancedOCRProcessor.

        Args:
            ocr_config (Optional[OCRConfig]): Configuration for OCR processing.
                If None, default OCRConfig settings are used. The configuration dictates aspects
                like page limits for PDFs, and whether to apply preprocessing steps.
        
        The constructor performs an initial check for Tesseract and Ollama vision model
        availability. This is done synchronously during init for simplicity, as it's a 
        one-time setup cost. For fully async applications, this check might be deferred or 
        handled by an async factory method.
        """
        self.config = ocr_config if ocr_config else OCRConfig()
        logger.info(f"Initializing AdvancedOCRProcessor with config: {self.config}")
        self.tesseract_available = False
        self.ollama_service_available = False # Renamed for clarity: refers to the Ollama service itself
        self.ollama_vision_model_available = False # Renamed: refers to the specific vision model
        
        # Run _check_dependencies_async in a blocking way during initialization.
        # This is acceptable here as it's a one-time setup cost for dependency checking.
        # For a fully event-loop driven application, one might use an async factory or 
        # ensure this is called within an existing loop.
        asyncio.run(self._check_dependencies_async())
        logger.info(
            f"OCR Dependency Status: Tesseract Available: {self.tesseract_available}, "
            f"Ollama Service Available: {self.ollama_service_available}, "
            f"Ollama Vision Model '{OLLAMA_VISION_MODEL_NAME}' Available: {self.ollama_vision_model_available}"
        )

    async def _check_dependencies_async(self):
        """
        Asynchronously checks for critical OCR dependencies: Tesseract and Ollama (service & vision model).
        
        This method verifies:
        1.  Tesseract Installation: Runs `tesseract --version` to confirm Tesseract is installed 
            and accessible in the system PATH. The rationale is that Tesseract is a common, reliable
            fallback OCR engine.
        2.  Ollama Service Reachability: Pings the Ollama `/api/tags` endpoint to ensure the 
            Ollama service (which serves local LLMs, including vision models) is running and responsive.
            This is crucial if Ollama-based OCR is intended.
        3.  Ollama Vision Model Availability: Checks if the specific vision model defined by
            `OLLAMA_VISION_MODEL_NAME` (from `app.config`) is listed by the Ollama service.
            This ensures the designated model for Ollama OCR is actually available.
            
        Updates internal flags (`self.tesseract_available`, `self.ollama_service_available`, 
        `self.ollama_vision_model_available`) based on these checks. These flags are then used 
        by other methods to determine which OCR engines can be attempted, enabling graceful
        degradation or selection of a primary engine.
        """
        # Check for Tesseract
        try:
            # Using asyncio.create_subprocess_exec for non-blocking subprocess call.
            process_tess = await asyncio.create_subprocess_exec(
                "tesseract", "--version", 
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process_tess.communicate() # Wait for command to complete
            if process_tess.returncode == 0:
                self.tesseract_available = True
                logger.info(f"Tesseract OCR is available. Version: {stdout.decode().splitlines()[0]}")
            else:
                # Log error from Tesseract if the command ran but failed.
                logger.warning(f"Tesseract --version command failed with code {process_tess.returncode}: {stderr.decode().strip()}")
                self.tesseract_available = False
        except FileNotFoundError:
            # This specific exception means the 'tesseract' executable was not found.
            logger.warning("Tesseract OCR command not found. Please ensure Tesseract is installed and in PATH.")
            self.tesseract_available = False
        except Exception as e:
            # Catch any other unexpected errors during Tesseract check.
            logger.error(f"An unexpected error occurred while checking Tesseract version: {e}", exc_info=True)
            self.tesseract_available = False

        # Check for Ollama service and specific vision model
        if not OLLAMA_BASE_URL or not OLLAMA_VISION_MODEL_NAME:
            logger.warning("Ollama base URL or vision model name is not configured in app.config. Ollama OCR will be unavailable.")
            self.ollama_service_available = False
            self.ollama_vision_model_available = False
            return # No point in proceeding with Ollama checks if not configured.
            
        try:
            # Use httpx.AsyncClient for asynchronous HTTP requests.
            async with httpx.AsyncClient(base_url=OLLAMA_BASE_URL, timeout=10.0) as client:
                # /api/tags is a standard Ollama endpoint to list all pulled/available models.
                response = await client.get("/api/tags") 
                if response.status_code == 200:
                    self.ollama_service_available = True
                    logger.info(f"Successfully connected to Ollama service at {OLLAMA_BASE_URL}.")
                    try:
                        models_data = response.json()
                        available_models = [model.get("name") for model in models_data.get("models", [])]
                        if OLLAMA_VISION_MODEL_NAME in available_models:
                            self.ollama_vision_model_available = True
                            logger.info(f"The configured Ollama vision model '{OLLAMA_VISION_MODEL_NAME}' is available.")
                        else:
                            self.ollama_vision_model_available = False
                            logger.warning(
                                f"Ollama vision model '{OLLAMA_VISION_MODEL_NAME}' NOT FOUND. "
                                f"Available models via Ollama API: {available_models}. "
                                f"Ensure the model is pulled (e.g., 'ollama pull {OLLAMA_VISION_MODEL_NAME}') and Ollama is running correctly."
                            )
                    except json.JSONDecodeError as e_json:
                        # If /api/tags doesn't return valid JSON, service might be up but unhealthy/misconfigured for this check.
                        logger.error(f"Failed to parse JSON response from Ollama /api/tags: {e_json}. Response text: {response.text[:200]}", exc_info=True)
                        self.ollama_service_available = True # Service responded, but tags endpoint is problematic
                        self.ollama_vision_model_available = False
                else:
                    # Service is reachable but returned an error status for /api/tags.
                    self.ollama_service_available = False # Or True, depending on interpretation of "available"
                    logger.warning(
                        f"Ollama service at {OLLAMA_BASE_URL} responded with status {response.status_code} for /api/tags. "
                        f"Ollama OCR may be unavailable. Response: {response.text[:200]}"
                    )
        except httpx.RequestError as e_http:
            # Covers connection errors, timeouts (if not status-related), etc.
            logger.warning(f"Could not connect to Ollama service at {OLLAMA_BASE_URL}: {e_http}. Ollama OCR will be unavailable.")
            self.ollama_service_available = False
            self.ollama_vision_model_available = False
        except Exception as e_general:
            # Catch-all for any other unexpected issues during Ollama check.
            logger.error(f"An unexpected error occurred while checking Ollama dependencies: {e_general}", exc_info=True)
            self.ollama_service_available = False
            self.ollama_vision_model_available = False

    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Applies preprocessing steps to a PIL Image to potentially enhance OCR accuracy.

        Current preprocessing involves:
        1.  Conversion to RGB (if not already, to ensure compatibility with OpenCV for grayscale conversion).
        2.  Conversion to grayscale (reduces complexity for OCR).
        3.  Application of Otsu's thresholding for binarization (creates a black and white image, \n            which can significantly help OCR by making text distinct from the background).
        
        These steps are common in OCR pipelines to standardize images and improve the signal-to-noise \n        ratio for text elements. The choice of Otsu's method is due to its adaptiveness in \n        finding an optimal threshold automatically.

        Args:
            image (Image.Image): The input PIL Image object.

        Returns:
            Image.Image: The preprocessed PIL Image. Returns the original image if \n                         an error occurs during preprocessing. This fallback ensures that if \n                         preprocessing fails (e.g., due to an unusual image format), the OCR \n                         process can still attempt to work with the original image, rather than failing entirely.
        """
        try:
            # Convert to RGB first to handle various modes like P (palette with limited colors),
            # RGBA (RGB with alpha/transparency), or L (luminance/grayscale already but maybe not in a way cv2 likes directly).
            # OpenCV\'s cvtColor expects a certain channel layout (e.g., BGR or RGB from a NumPy array).
            if image.mode != 'RGB':
                img_rgb = image.convert('RGB')
            else:
                img_rgb = image
            
            img_array = np.array(img_rgb) 
            # Convert the RGB image to grayscale.
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Apply Otsu\'s thresholding to binarize the image.
            # This automatically determines an optimal global threshold value from the image histogram.
            # It's particularly effective for bimodal histograms (e.g., dark text on light background).
            # cv2.THRESH_BINARY: Pixels above threshold become 255 (white), below become 0 (black).
            # cv2.THRESH_OTSU: Flag to enable Otsu\'s method.
            _, binary_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            logger.debug("Image preprocessed successfully (Converted to RGB -> Grayscale -> Otsu\'s Thresholding)." ) # Added detail
            return Image.fromarray(binary_img)
        except Exception as e:
            logger.error(f"Error during image preprocessing: {e}. Returning original image.", exc_info=True)
            return image # Fallback to original image on error

    def _convert_pdf_to_images(self, pdf_input: Union[str, bytes]) -> List[Image.Image]:
        """
        Converts a PDF document (from a file path or bytes) into a list of PIL Image objects.

        This method is essential for processing PDF documents, as OCR engines operate on images, 
        not directly on PDF vector/text data. It uses the `pdf2image` library, which acts as a 
        Python wrapper around the Poppler PDF rendering utilities (specifically `pdftoppm`). 
        Therefore, Poppler must be installed on the system for this to work.

        It respects the `max_pages` and `skip_first_n_pages` settings from the `OCRConfig` 
        to allow selective processing of PDF pages. This is useful for handling large PDFs or 
        skipping irrelevant cover/appendix pages.

        Args:
            pdf_input (Union[str, bytes]): Path to the PDF file (str) or PDF content as bytes.
                                           Providing bytes is useful for in-memory processing without disk I/O.

        Returns:
            List[Image.Image]: A list of PIL Image objects, one for each selected and successfully converted page.
                               Returns an empty list if no pages are converted (e.g., empty PDF, invalid range, or error).

        Raises:
            FileNotFoundError: If `pdf_input` is a path (str) and the file does not exist.
            TypeError: If `pdf_input` is not of type str or bytes.
            RuntimeError: If PDF conversion fails. This typically wraps errors from `pdf2image` such as \n                          `PDFInfoNotInstalledError` (Poppler not found or not in PATH), \n                          `PDFPageCountError` (issues determining page count), or \n                          `PDFSyntaxError` (malformed PDF). Detailed error from Poppler is logged.
        """
        first_page_to_process = self.config.skip_first_n_pages + 1 # pdf2image is 1-indexed for pages
        last_page_to_process = None
        if self.config.max_pages > 0:
            last_page_to_process = self.config.skip_first_n_pages + self.config.max_pages

        logger.info(f"Converting PDF to images. Effective page range for conversion: {first_page_to_process} to {last_page_to_process if last_page_to_process else 'end'}. Timeout set to 120s.")
        
        images: List[Image.Image] = []
        try:
            if isinstance(pdf_input, str):
                if not os.path.exists(pdf_input):
                    logger.error(f"PDF file not found at path: {pdf_input}")
                    raise FileNotFoundError(f"PDF file not found: {pdf_input}")
                # `convert_from_path` uses Poppler\'s pdftoppm utility.
                # A timeout is important for very large or complex PDFs that might hang pdftoppm.
                images = convert_from_path(
                    pdf_input, 
                    first_page=first_page_to_process, 
                    last_page=last_page_to_process, 
                    timeout=120, # Increased from 60s to 120s for more robustness
                    # dpi=200, # Default is 200, can be adjusted if needed for resolution vs size/speed trade-off
                    # thread_count=4 # Can parallelize page conversion if Poppler build supports it
                )
            elif isinstance(pdf_input, bytes):
                images = convert_from_path(
                    None, # No path needed when using pdf_bytes
                    pdf_bytes=pdf_input, 
                    first_page=first_page_to_process, 
                    last_page=last_page_to_process, 
                    timeout=120 # Increased from 60s to 120s
                )
            else:
                # This check ensures the input conforms to what the method expects.
                raise TypeError(f"pdf_input must be a file path (str) or bytes, but got {type(pdf_input)}.")
            
            if not images:
                # This can happen if the PDF is empty, has no pages in the selected range, or is corrupted in a way 
                # that pdf2image doesn\'t error but returns no images.
                logger.warning("PDF conversion resulted in no images. This could be due to an empty PDF, an invalid page range (e.g., skip_first_n_pages exceeding total pages), or a corrupted PDF not caught by Poppler errors.")

        except (PDFInfoNotInstalledError, PDFPageCountError, PDFSyntaxError) as e:
            # These are specific exceptions from pdf2image, often indicating Poppler issues.
            logger.error(
                f"pdf2image specific error: {e}. This usually means Poppler PDF rendering utilities "
                f"(pdftoppm, pdfinfo) are NOT INSTALLED or NOT FOUND in the system PATH. "
                f"On Debian/Ubuntu: 'sudo apt-get install poppler-utils'. On macOS: 'brew install poppler'. "
                f"On Windows: Download Poppler binaries, extract, and add the \'bin\' directory to PATH.", 
                exc_info=False # Error message is already very descriptive
            )
            # Raising a RuntimeError makes it clear to the caller that a fundamental step failed.
            raise RuntimeError(f"PDF conversion failed due to missing or problematic Poppler installation: {type(e).__name__} - {e}") from e
        except Exception as e:
            # Catch any other unexpected errors during conversion.
            logger.error(f"An unexpected error occurred during PDF to image conversion: {e}", exc_info=True)
            raise RuntimeError(f"General PDF conversion failed: {type(e).__name__} - {e}") from e
        
        logger.info(f"Successfully converted PDF to {len(images)} image(s) for OCR.")
        return images

    async def _ocr_image_with_tesseract_async(self, image: Image.Image, lang: str = "eng") -> str:
        """
        Asynchronously performs OCR on a single PIL Image using the Tesseract OCR engine.

        This method is a core part of the OCR strategy, providing a reliable open-source OCR capability.
        It requires Tesseract to be installed on the system and accessible via the system PATH.
        The method works by:
        1.  Saving the input PIL Image to a temporary PNG file (PNG is chosen as a lossless format \n            suitable for OCR).
        2.  Invoking the `tesseract` command-line tool as an asynchronous subprocess, pointing it \n            to the temporary image file and requesting output to stdout.
        3.  Specifying the language (defaulting to English, `eng`). Accurate language specification is \n            critical for Tesseract's performance.
        4.  Using Page Segmentation Mode (PSM) 3, which is Tesseract's default for fully automatic \n            page segmentation, often a good balance for general documents.
        5.  Capturing and decoding Tesseract's stdout (the extracted text) and stderr (for error messages).
        6.  Cleaning up the temporary file.

        Args:
            image (Image.Image): The PIL Image object to perform OCR on.
            lang (str): The language code for Tesseract (e.g., "eng" for English, "deu" for German). \n                        Defaults to "eng". It is crucial that the corresponding language data file \n                        (e.g., `eng.traineddata`) is available in Tesseract's `tessdata` directory.\n                        Using the wrong language or missing language data will lead to poor or no results.

        Returns:\n            str: The extracted text. Returns an empty string if Tesseract is unavailable (as determined \n                 during initialization), if the Tesseract process fails (e.g., non-zero exit code), \n                 or if no text is effectively found.
        """
        if not self.tesseract_available:
            logger.warning("Tesseract OCR engine is not available (based on initial checks). Skipping Tesseract OCR attempt.")
            return ""
        
        # Using a context manager for the temporary file is generally safer, but since we need the path 
        # after the `with` block for the subprocess and then need to delete it manually in `finally`, 
        # direct NamedTemporaryFile usage with manual deletion is employed here.
        # delete=False is necessary, especially on Windows, to allow the subprocess to access the file by name.
        tmp_file_path = None # Initialize to ensure it's defined for the finally block
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                image.save(tmp_file.name, format="PNG") # Save in a lossless format like PNG
                tmp_file_path = tmp_file.name
            
            # Construct Tesseract command. 
            # "stdout" tells Tesseract to output text to standard output.
            # "-l <lang>" specifies the language.
            # "--psm 3" (Page Segmentation Mode 3) is Tesseract's default for fully automatic page segmentation.
            # Other PSM modes could be exposed via OCRConfig if more control is needed for specific document types.
            cmd = ["tesseract", tmp_file_path, "stdout", "-l", lang, "--psm", "3"]
            logger.info(f"Executing Tesseract OCR for language '{lang}' with PSM 3 on temporary image: {tmp_file_path}")
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE, 
                stderr=asyncio.subprocess.PIPE
            )
            stdout_bytes, stderr_bytes = await process.communicate() # Wait for Tesseract to finish
            
            if process.returncode == 0:
                extracted_text = stdout_bytes.decode("utf-8").strip()
                logger.info(f"Tesseract OCR successful for image. Extracted {len(extracted_text)} characters.")
                return extracted_text
            else:
                # Tesseract often provides useful error messages on stderr.
                error_message = stderr_bytes.decode("utf-8").strip()
                logger.error(f"Tesseract OCR failed for image {tmp_file_path} with return code {process.returncode}. Error: {error_message}")
                return "" # Return empty string on Tesseract error
        except Exception as e:
            # Catch any other unexpected errors during the Tesseract process (e.g., issues saving temp file).
            logger.error(f"An unexpected error occurred during Tesseract OCR processing: {e}", exc_info=True)
            return "" # Return empty string on unexpected errors
        finally:
            # Ensure the temporary file is always cleaned up, regardless of success or failure.
            if tmp_file_path and os.path.exists(tmp_file_path):
                try:
                    os.remove(tmp_file_path)
                    logger.debug(f"Successfully removed temporary OCR image file: {tmp_file_path}")
                except Exception as e_rem:
                    logger.warning(f"Failed to remove temporary OCR image file {tmp_file_path}: {e_rem}")

    async def _ocr_image_with_ollama_async(self, image: Image.Image, ollama_model_name: str = OLLAMA_VISION_MODEL_NAME, prompt: Optional[str] = None) -> str:
        """
        Asynchronously performs OCR on a single PIL Image using a specified Ollama vision model 
        via its HTTP API.

        This method allows leveraging local vision LLMs (e.g., LLaVA, Moondream, BakLLaVA) served by Ollama \n        for OCR tasks. It offers an alternative to traditional OCR engines like Tesseract and can be \n        particularly useful for documents where contextual understanding might aid transcription, or for \n        models specifically fine-tuned for OCR or document understanding tasks.

        The process involves:
        1.  Checking if the Ollama service and the specific vision model are marked as available (from init checks).
        2.  Converting the PIL Image to PNG format and then to base64 encoded bytes, as required by the \n            Ollama `/api/generate` endpoint for multimodal inputs.
        3.  Constructing a JSON payload containing the model name, a prompt (defaulting to simple \n            transcription if not provided), and the base64 encoded image(s).
        4.  Making an asynchronous POST request to the Ollama API.
        5.  Parsing the JSON response to extract the transcribed text.
        6.  Includes error handling for API request failures, HTTP errors, and JSON parsing issues.

        Args:
            image (Image.Image): The PIL Image object to perform OCR on.
            ollama_model_name (str): The name of the Ollama vision model to use (e.g., "llava:latest", \n                                     "moondream:latest"). This should match a model available in the \n                                     local Ollama instance. Defaults to `OLLAMA_VISION_MODEL_NAME` from config.
            prompt (Optional[str]): An optional specific prompt to guide the vision model. 
                                   If None, a default general-purpose transcription prompt is used. \n                                   Custom prompts can be used to tailor the OCR for specific layouts or \n                                   to instruct the model to focus on certain regions or text types, \n                                   though the effectiveness varies by model.

        Returns:\n            str: The extracted text from the Ollama model's response. Returns an empty string \n                 if Ollama is unavailable, the model is not found, the API call fails, \n                 no text is extracted, or any other error occurs during the process.
        """
        if not self.ollama_service_available:
            logger.warning("Ollama service is not available (based on initial checks). Skipping Ollama OCR attempt.")
            return ""
        
        # Check for the specific model passed as argument, or the default configured one if they match.
        # If a custom model is passed, we rely on Ollama to report if it's not found.
        # If it's the default configured model, and we know it's unavailable from init checks, we can skip early.
        if ollama_model_name == OLLAMA_VISION_MODEL_NAME and not self.ollama_vision_model_available:
            logger.warning(f"Default Ollama vision model '{OLLAMA_VISION_MODEL_NAME}' is not available (based on initial checks). Skipping Ollama OCR.")
            return ""

        try:
            # Convert image to PNG bytes and then base64 encode. PNG is a good lossless format.
            with tempfile.NamedTemporaryFile(suffix=".png") as tmp_img_file:
                image.save(tmp_img_file.name, format="PNG") 
                with open(tmp_img_file.name, "rb") as f:
                    image_bytes = f.read()
            base64_image = base64.b64encode(image_bytes).decode("utf-8")
        except Exception as e_img_conv:
            logger.error(f"Error converting image to base64 for Ollama model '{ollama_model_name}': {e_img_conv}", exc_info=True)
            return ""

        # Use a default prompt if none is provided, guiding the model for general transcription.
        # For more complex tasks (e.g., Visual Question Answering), a more specific prompt would be necessary.
        current_prompt = prompt if prompt else "Describe the image in detail. Focus on transcribing any text visible in the image accurately. Output only the transcribed text."
        
        payload = {
            "model": ollama_model_name,
            "prompt": current_prompt,
            "images": [base64_image], # Ollama API for multimodal models expects a list of base64 encoded images
            "stream": False  # We want the full response at once for OCR, not a streaming response
        }
        
        # Use the global Ollama request timeout from app.config, ensuring it's a float.
        # Add a small buffer (e.g., 10s) for the AsyncClient itself if the main timeout is for the request body processing.
        configured_timeout = float(os.getenv("OLLAMA_REQUEST_TIMEOUT", "600")) 

        try:
            async with httpx.AsyncClient(base_url=OLLAMA_BASE_URL, timeout=configured_timeout + 10.0) as client:
                logger.info(f"Sending OCR request to Ollama model '{ollama_model_name}'. Image size (bytes before base64): {len(image_bytes)}. Prompt (first 100 chars): '{current_prompt[:100]}...'")
                # The request itself also uses the configured_timeout for the actual POST operation.
                response = await client.post("/api/generate", json=payload, timeout=configured_timeout) 
                response.raise_for_status() # Raises HTTPStatusError for 4xx/5xx responses, simplifying error checking.
                
                response_data = response.json()
                extracted_text = response_data.get("response", "").strip() # 'response' is the key Ollama uses for the generated text.
                
                # Simple heuristic to clean common markdown-like artifacts if the model adds them unnecessarily for plain text.
                # This can happen if the model is trying to format its output as code or a specific text block.
                extracted_text = re.sub(r"^```(?:json|text)?\\n?|^```\\n?|\\n?```$", "", extracted_text, flags=re.MULTILINE).strip()

                if not extracted_text and response_data:
                     # Log if API call was successful (200 OK) but no text was returned, which might indicate an issue with the prompt or model for that image.
                     logger.warning(f"Ollama OCR with model '{ollama_model_name}' returned empty text despite a successful API response. Full response (first 500 chars for brevity): {str(response_data)[:500]}")
                elif extracted_text:
                    logger.info(f"Ollama OCR with model '{ollama_model_name}' successful. Extracted text length: {len(extracted_text)} characters.")
                # If extracted_text is empty and response_data was also empty (shouldn't happen on 200 OK), no specific log here, it's just an empty result.
                return extracted_text

        except httpx.HTTPStatusError as e_status:
            # Handles errors where the Ollama server responded with an HTTP error code (4xx or 5xx).
            error_body = ""
            try:
                error_body = e_status.response.text # Attempt to get more details from response body
            except Exception: 
                pass # Keep error_body empty if it can't be read for some reason
            
            logger.error(
                f"Ollama API HTTP error (status {e_status.response.status_code}) for model '{ollama_model_name}'. "
                f"URL: {e_status.request.url}. Response (first 500 chars): {error_body[:500]}", 
                exc_info=False # Typically, the status and body are enough, exc_info=True can be very verbose for HTTP errors.
            )
            # Specifically check for "model not found" type errors, which are common.
            if "model not found" in error_body.lower() or e_status.response.status_code == 404:
                 logger.error(
                    f"The Ollama model '{ollama_model_name}' was not found by the Ollama service. "
                    f"Please ensure it is pulled (e.g., 'ollama pull {ollama_model_name}') and that the Ollama service is running correctly and has loaded the model."
                )
            return "" # Return empty string on HTTP error
        except httpx.RequestError as e_req:
            # Covers network errors (e.g., connection refused, DNS resolution failure), or timeouts not caught by HTTPStatusError.
            logger.error(f"Request to Ollama API failed for model '{ollama_model_name}': {e_req}. URL: {e_req.request.url if e_req.request else 'N/A'}", exc_info=True)
            return "" # Return empty string on request error
        except json.JSONDecodeError as e_json_dec:
            # This occurs if the server responds with 200 OK but the body is not valid JSON.
            response_text_sample = "Response object not available"
            if 'response' in locals() and hasattr(response, 'text'):
                response_text_sample = response.text[:500] # Get sample of non-JSON response
            logger.error(f"Failed to decode JSON response from Ollama for model '{ollama_model_name}'. Response text sample: {response_text_sample}", exc_info=True)
            return "" # Return empty string on JSON decode error
        except Exception as e_unexpected: # Catch any other unexpected errors during the Ollama API call or response processing.
            logger.error(f"An unexpected error occurred during Ollama OCR with model '{ollama_model_name}': {e_unexpected}", exc_info=True)
            return "" # Return empty string on other unexpected errors

    async def _extract_text_from_single_image_async(self, image: Image.Image, use_ollama_first: bool) -> str:
        """
        Asynchronously extracts text from a single preprocessed image, strategically choosing the OCR engine.

        This method implements the core logic for deciding which OCR engine to use and handling fallbacks.
        The strategy is:
        1.  If `use_ollama_first` is True AND the Ollama service is available AND the configured \n            Ollama vision model (`OLLAMA_VISION_MODEL_NAME`) is available (all based on prior checks):
            - Attempt OCR using `_ocr_image_with_ollama_async` with the default vision model.
            - If successful (returns non-empty text), return this text immediately.
            - If it fails or returns empty text, log a warning and proceed to Tesseract fallback (if available).
        2.  If Tesseract is available (and either Ollama wasn't tried, wasn't available, or failed):
            - Attempt OCR using `_ocr_image_with_tesseract_async`.
            - If successful, return this text.
            - If Tesseract also fails or returns empty text, log a warning.
        3.  If neither Ollama nor Tesseract yields text (either due to unavailability or failure), \n            log a warning and return an empty string.

        The `use_ollama_first` parameter allows the caller (ultimately driven by application \n        configuration or specific needs) to prioritize the potentially more advanced but perhaps \n        slower/less reliable Ollama vision models, or to default to the typically faster/more \n        robust Tesseract engine.

        Args:
            image (Image.Image): The preprocessed PIL Image object to extract text from.
            use_ollama_first (bool): If True, try the configured Ollama vision model first. \n                                   Otherwise, Tesseract is prioritized if Ollama is not selected or fails.

        Returns:\n            str: The extracted text. Returns an empty string if all attempted OCR methods fail, \n                 are unavailable, or return no usable text.
        """
        extracted_text = ""
        
        # Primary attempt: Ollama Vision Model, if prioritized and available
        if use_ollama_first and self.ollama_service_available and self.ollama_vision_model_available:
            logger.info(f"Attempting OCR with configured Ollama vision model: '{OLLAMA_VISION_MODEL_NAME}'.")
            extracted_text = await self._ocr_image_with_ollama_async(image, ollama_model_name=OLLAMA_VISION_MODEL_NAME)
            if extracted_text: # Check if Ollama returned any text
                logger.info(f"Successfully extracted text using Ollama model '{OLLAMA_VISION_MODEL_NAME}'. Length: {len(extracted_text)}.")
                return extracted_text # Return immediately if Ollama succeeds
            else:
                # Log that Ollama attempt failed or returned nothing, before trying Tesseract.
                logger.warning(f"Ollama OCR attempt with model '{OLLAMA_VISION_MODEL_NAME}' failed or returned empty text. Attempting Tesseract fallback if available.")
        
        # Fallback or Primary attempt: Tesseract OCR
        if self.tesseract_available:
            if use_ollama_first and not extracted_text: # Only log explicit fallback if Ollama was tried and failed to produce text
                logger.info("Falling back to Tesseract OCR after Ollama attempt yielded no text.")
            elif not use_ollama_first:
                logger.info("Attempting OCR with Tesseract (Ollama not prioritized or not configured/available for this attempt)." )
            # If use_ollama_first was true but ollama service/model wasn't available, this path is also taken, 
            # but the log above (if ollama failed) or no specific log (if ollama unavailable) is sufficient.
            
            tesseract_text = await self._ocr_image_with_tesseract_async(image)
            if tesseract_text: # Check if Tesseract returned any text
                logger.info(f"Successfully extracted text using Tesseract. Length: {len(tesseract_text)}.")
                return tesseract_text # Return Tesseract's text
            else:
                # Log that Tesseract also failed or returned nothing.
                logger.warning("Tesseract OCR attempt also failed or returned empty text.")
        
        # If we reach here, neither Ollama (if tried) nor Tesseract produced usable text, or Tesseract wasn't available.
        if not extracted_text: # This condition will be true if Ollama wasn't tried or failed, AND Tesseract failed or wasn't available.
             logger.warning(
                f"No OCR engines (Ollama prioritized: {use_ollama_first}, Tesseract available: {self.tesseract_available}) "
                f"succeeded in extracting text from the image or were available."
            )

        return extracted_text # This will be an empty string if all attempts failed.

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

