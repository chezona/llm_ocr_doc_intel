"""
This module utilizes LiteLLM and the Instructor library to process text content (typically from OCR)
with an Ollama-compatible LLM. It aims to classify the document type (e.g., PSEG bill,
general invoice, receipt) and extract structured information based on predefined Pydantic models.
"""
import instructor
import litellm
from pydantic import ValidationError
import json # Standard library for potential JSON manipulation if needed, though instructor handles main parsing.
from typing import Tuple # For type hinting return types.
import traceback # For detailed error logging in except blocks.

from app.models import ExtractedData # Main data structure for final output.
# Specific data models (PSEGData, InvoiceData, Receipt) are used by LLMResponseModel.
from app.config import OLLAMA_BASE_URL, OLLAMA_EXTRACTOR_MODEL_NAME, OLLAMA_REQUEST_TIMEOUT
from app.llm_response_models import LLMResponseModel, SimpleClassificationOutput # Added SimpleClassificationOutput

# --- LiteLLM Global Configuration --- # 
# Set the base URL for Ollama as a custom OpenAI-compatible endpoint for LiteLLM.
# This allows LiteLLM to route requests to the correct Ollama service.
litellm.api_base = OLLAMA_BASE_URL

# --- Instructor Client Initialization --- #
# Create an Instructor client that wraps LiteLLM's completion function.
# mode=instructor.Mode.JSON ensures that Instructor attempts to parse the LLM's output
# directly into the specified Pydantic `response_model` and handles retries on validation errors.
instructor_litellm_client = instructor.from_litellm(litellm.completion, mode=instructor.Mode.JSON)

# Function to get the Ollama model identifier
def _get_ollama_model_identifier(model_name: str) -> str:
    """
    Ensures the Ollama model name is formatted correctly for LiteLLM.

    LiteLLM expects Ollama model names to be prefixed with "ollama/" 
    (e.g., "ollama/mistral:7b-instruct-q4_K_M"). This function adds the 
    prefix if it's missing.

    Args:
        model_name: The raw model name (e.g., "mistral:7b-instruct-q4_K_M").

    Returns:
        The model name, prefixed with "ollama/" if it wasn't already.
    """
    if not model_name.startswith("ollama/"):
        return f"ollama/{model_name}"
    return model_name

def classify_document_type(text: str, model_name: str) -> SimpleClassificationOutput:
    """
    Classifies the document type using an LLM and a focused prompt.

    Args:
        text: The input text (OCR output) to classify.
        model_name: The Ollama model name to use for classification.

    Returns:
        A SimpleClassificationOutput object containing the classified document type
        and an optional confidence score.
    """
    classification_prompt = """
# Document Classification Prompt

You are an expert document analysis AI. Your task is to classify the following text into one of these categories: 'pseg_bill', 'invoice', 'receipt', or 'other'.
Consider the overall content, keywords, and structure.

**Priority is to correctly identify 'pseg_bill'.**

Supported document types:
- **'pseg_bill'**: Utility bills from PSE&G (Public Service Electric and Gas). Look for terms like "PSEG", "Public Service Electric and Gas", "Account Number", "Bill Date", "Amount Due", and a layout typical of utility bills.
  *Example of text indicating a PSEG bill: "Your PSEG Long Island Bill ... Account Number ... Total Amount Due" -> classify as "pseg_bill".*
- 'invoice': General invoices for goods or services. These usually have "Invoice #", "Vendor", "Client", "Line Items", "Subtotal", "Total".
- 'receipt': Proof of purchase from retail or services. Typically show "Merchant Name", "Transaction Date", "Items Purchased", "Total Paid".
- 'other': Any document not fitting the above categories.

Focus on identifying key characteristics of a PSEG utility bill. These often include:
- The exact words "PSEG" or the phrase "Public Service Electric and Gas".
- Phrases like "Your PSEG Long Island Bill", "Amount Due", "Bill Date", "Account Number", "Usage Details", "Payment Information".
- A structure resembling a utility statement with charges, meter readings (if present), and service addresses.

If the text contains strong indicators of a PSEG bill (e.g., mentions "PSEG" AND an "Account Number" AND "Amount Due"), you **must** classify it as 'pseg_bill'.

Return your analysis *only* as a single, valid JSON object with the following structure:
```json
{{
  "document_type": "pseg_bill", // or "invoice", "receipt", "other"
  "confidence": 0.9 // 0.0 to 1.0
}}
```

Input Text to Analyze:
{{{{text_content}}}}
    """

    print(f"Classifying document type with model: {model_name}. Text (first 200 chars): {text[:200]}....")
    
    ollama_model_identifier = _get_ollama_model_identifier(model_name)

    try:
        response: SimpleClassificationOutput = instructor_litellm_client.chat.completions.create(
            model=ollama_model_identifier,
            messages=[
                {"role": "user", "content": classification_prompt.format(text_content=text)}
            ],
            response_model=SimpleClassificationOutput,
            max_retries=1,
            api_base=OLLAMA_BASE_URL,
            request_timeout=OLLAMA_REQUEST_TIMEOUT # Consider a shorter timeout for classification if it's faster
        )
        print(f"Classification successful: {response}")
        return response
    except Exception as e:
        print(f"Error during document classification: {e}")
        print(traceback.format_exc())
        # Fallback to 'other' in case of classification error
        return SimpleClassificationOutput(document_type="other", confidence=0.0)

class DocumentProcessor:
    """
    Handles LLM interaction for document classification and structured data extraction.
    Uses the `instructor` library wrapping `litellm` to achieve robust JSON output
    from an Ollama-compatible LLM, fitting into a Pydantic model.
    """
    def __init__(self, model_name: str):
        """
        Initializes the DocumentProcessor.

        Args:
            model_name: The name of the Ollama model to be used (e.g., "mistral:7b-instruct-q4_K_M").
                        This is typically passed from the application configuration or calling function.
        """
        self.model_name = model_name
        # Note: The OLLAMA_BASE_URL is configured globally for litellm via `litellm.api_base`.
        # The OLLAMA_REQUEST_TIMEOUT is passed directly in the `create` call.

        # The extensive prompt template guides the LLM in classifying the document
        # and extracting information according to the Pydantic schemas defined in LLMResponseModel.
        self.prompt_template = '''
# PSEG Bill Data Extraction Prompt

You are an expert data extraction system. The input text has been identified as a **PSEG utility bill**.
Your task is to extract structured information from this PSEG utility bill.
The input text is from OCR and may contain errors or irrelevant information.

Extract all available information for these fields:
- **account_number**: The PSEG account identifier (e.g., "7730566702").
- **customer_name**: Full name of the account holder (e.g., "NWANKWO OKECHUKWU.A.").
- **service_address**: Complete service address including unit/floor (e.g., "167 SCHEERER AVE FL 2, NEWARK CITY NJ 07112-2113").
- **billing_address**: Mailing address if different from service address (if same or not present, can be null or repeat service_address).
- **bill_date**: Date the bill was generated (YYYY-MM-DD format, e.g., "2024-04-12").
- **billing_period_start**: Start date of billing cycle (YYYY-MM-DD format, e.g., "2024-04-02").
- **billing_period_end**: End date of billing cycle (YYYY-MM-DD format, e.g., "2024-04-10").
- **total_amount_due**: Total monetary amount to be paid (as a number, e.g., 765.41).
- **due_date**: Payment due date (YYYY-MM-DD format, e.g., "2024-04-29").
- **previous_balance**: Balance from the previous bill (as a number, e.g., 0.00).
- **current_charges**: Sum of current charges and credits for the billing period (as a number, e.g., 765.41).
- **next_meter_reading**: Date for the next scheduled meter reading (YYYY-MM-DD format, e.g., "2024-05-09", if available).
- **utility_company**: Name of the utility company (should be "PSE&G" or similar).
- **customer_service_phone**: Customer service phone number (e.g., "1-800-436-PSEG").
- **payment_methods**: Brief description or list of available payment options mentioned.

**PSEG Bill Example:**

Input Text Snippet:
"""
PSE&G
Account Number: 7730566702
Bill Date: April 12, 2024 Due Date: April 29, 2024
NWANKWO OKECHUKWU.A.
Service For: 167 SCHEERER AVE FL 2, NEWARK CITY NJ 07112-2113
Total Amount Due: $765.41
Billing Period: 04/02/2024 - 04/10/2024
"""

Desired JSON Output for the above snippet:
```json
{{
  "document_type": "pseg_bill",
  "structured_data": {{
    "account_number": "7730566702",
    "customer_name": "NWANKWO OKECHUKWU.A.",
    "service_address": "167 SCHEERER AVE FL 2, NEWARK CITY NJ 07112-2113",
    "billing_address": null,
    "bill_date": "2024-04-12",
    "billing_period_start": "2024-04-02",
    "billing_period_end": "2024-04-10",
    "total_amount_due": 765.41,
    "due_date": "2024-04-29",
    "previous_balance": null,
    "current_charges": null, // Assuming not in snippet
    "next_meter_reading": null,
    "utility_company": "PSE&G",
    "customer_service_phone": null,
    "payment_methods": null
  }}
}}
```

## Output Format and Extraction Rules:
1. Your response **MUST** be a single, valid JSON object. Do not include any explanatory text, markdown formatting, or any characters before or after the JSON block.
2. The top-level keys in the JSON object **MUST** be "document_type" and "structured_data".
3. The "document_type" **MUST** be "pseg_bill" for this task.
4. For the "structured_data" object, you **MUST** attempt to extract a value for every field listed in the "Extract all available information for these fields" section above.
5. If, after careful and thorough analysis of the entire input text, the information for a specific field is unequivocally absent or impossible to determine, and **only in that case**, use the JSON value `null` for that field.
6. Do **NOT** return an empty "structured_data" object (e.g., `{{}}`) or an object where all field values are `null` if the relevant information is present in the input text. Your goal is to populate "structured_data" as completely and accurately as possible.
7. Ensure all date fields are in YYYY-MM-DD format. Ensure monetary amounts are numbers, not strings with currency symbols.

Input Text to Analyze:
{{{{text_content}}}}
'''

    def process_text(self, text: str) -> tuple[str, dict]:
        """
        Processes the input text using the configured LLM to classify its type
        and extract structured data based on the `LLMResponseModel`.

        Args:
            text: The raw text content (typically from OCR) to be processed.

        Returns:
            A tuple containing:
                - doc_type (str): The classified document type ('pseg_bill', 'invoice', 'receipt', 'other').
                - structured_data (dict): A dictionary containing the extracted data relevant to the doc_type.
                                          For 'other', this will be a dict like {"note": "..."}.
                                          In case of processing errors, this will contain an {"error": "..."} structure.
        """
        print(f"Processing text with litellm and instructor. Model: {self.model_name}. Text (first 200 chars): {text[:200]}....")

        # Ensure the model identifier for litellm/Ollama is correctly formatted.
        # LiteLLM expects "ollama/model_name" for Ollama models when calling generic completion.
        ollama_model_identifier = _get_ollama_model_identifier(self.model_name)
        
        try:
            # Make the call to the LLM via LiteLLM, with Instructor handling response parsing and retries.
            # - model: Specifies the Ollama model to use (e.g., "ollama/mistral:7b-instruct-q4_K_M").
            # - response_model: The Pydantic model (LLMResponseModel) Instructor should parse the JSON into.
            # - max_retries: Number of times Instructor will retry if Pydantic validation fails or LLM output is malformed.
            # - api_base: Explicitly passed to ensure LiteLLM routes to the correct Ollama instance.
            # - request_timeout: Timeout for this specific LLM request, from app.config.
            response: LLMResponseModel = instructor_litellm_client.chat.completions.create(
                model=ollama_model_identifier, 
                messages=[
                    {"role": "user", "content": self.prompt_template.format(text_content=text)}
                ],
                response_model=LLMResponseModel, 
                max_retries=2, # Configurable: number of retries for parsing/validation errors by Instructor.
                api_base=OLLAMA_BASE_URL, # Ensures LiteLLM targets the correct Ollama service.
                request_timeout=OLLAMA_REQUEST_TIMEOUT # Timeout for the LLM call itself.
            )
            
            print(f"LLM call successful. Parsed Response (JSON): {response.model_dump_json(indent=2)}")
            print(f"LLM call successful. Response: {response}")
            return response.document_type, response.structured_data

        except litellm.exceptions.APIConnectionError as e:
            # Handles errors where LiteLLM cannot connect to the Ollama service.
            print(f"LiteLLM API Connection Error: Could not connect to Ollama at {OLLAMA_BASE_URL}. Error: {e}")
            print(traceback.format_exc())
            return "other", {"error": f"LiteLLM API Connection Error: {str(e)}"}
        except instructor.exceptions.InstructorRetryException as e:
            # Handles errors from Instructor after exhausting retries (e.g., persistent validation failures).
            print(f"Instructor Retry Exception after {e.retries} attempts. Error: {e}")
            raw_llm_output_on_failure = "Raw LLM output not available."
            if hasattr(e, 'last_completion') and e.last_completion and e.last_completion.choices:
                raw_llm_output_on_failure = e.last_completion.choices[0].message.content
                print(f"Raw LLM response on failure: {raw_llm_output_on_failure}")
            elif hasattr(e, 'body') and e.body: # For Pydantic validation errors wrapped by Instructor
                 # This typically contains the data that failed Pydantic validation, which might be the raw response
                 raw_llm_output_on_failure = str(e.body)
                 print(f"Data body that failed Pydantic validation: {raw_llm_output_on_failure}")
            
            print(traceback.format_exc())
            # Include the actual raw output in the error dictionary for the workflow if possible
            return "other", {"error": f"Instructor Retry Exception after {e.retries} retries: {str(e)}", "raw_llm_output_snippet": raw_llm_output_on_failure[:1000]} # Increased snippet length
        except ValidationError as e: # Pydantic's own ValidationError
            # This might be caught if Instructor fails to catch it or if validation happens outside Instructor.
            print(f"Pydantic ValidationError: The LLM response did not match the LLMResponseModel schema. Errors: {e.errors()}")
            raw_response_body = "Raw response body not directly available in Pydantic ValidationError."
            # Try to get the JSON body if the exception has it (more common with FastAPI/Pydantic request validation)
            if hasattr(e, 'json') and callable(e.json):
                try:
                    raw_response_body = e.json()
                except Exception:
                    pass # Ignore if .json() fails
            print(f"Raw data causing validation error (if available): {raw_response_body}")
            print(traceback.format_exc())
            return "other", {"error": f"Pydantic ValidationError: {str(e)}", "validation_errors": e.errors()}
        except Exception as e:
            # Catch-all for any other unexpected errors during LLM processing.
            print(f"General error during LLM processing with litellm/instructor. Error type: {type(e)}, Error: {str(e)}")
            raw_response_data = "Raw response data not directly extractable from this general exception."
            if hasattr(e, 'response') and hasattr(e.response, 'text'): # For errors from HTTP libraries like requests/httpx
                raw_response_data = e.response.text
            elif hasattr(e, 'args') and len(e.args) > 0:
                # Sometimes the failing data is in args[0], e.g. JSON parsing errors from instructor/Pydantic
                if isinstance(e.args[0], str) and ("{" in e.args[0] or "[" in e.args[0]):
                    raw_response_data = e.args[0]
            print(f"Raw response or error details (if found): {raw_response_data[:1000] if isinstance(raw_response_data, str) else raw_response_data}")
            print(traceback.format_exc())
            return "other", {"error": f"General LLM processing error: {str(e)}", "details": raw_response_data[:500] if isinstance(raw_response_data, str) else str(raw_response_data)}

def process_ocr_text_with_llm(ocr_text: str, model_name: str) -> ExtractedData:
    """
    Processes OCR'd text to classify its type and extract structured information
    using a specified LLM model. This function is intended for detailed extraction
    once a document type is potentially known (e.g., for PSEG bills).

    Args:
        ocr_text: The text extracted from a document by an OCR process.
        model_name: The Ollama model name to use for this processing step.

    Returns:
        An ExtractedData object containing the classification, structured information,
        and other metadata.
    """
    # Instantiate the processor with the provided model_name
    # OLLAMA_EXTRACTOR_MODEL_NAME is no longer used here directly; model is passed.
    processor = DocumentProcessor(model_name=model_name)
    
    doc_type, structured_info = processor.process_text(ocr_text)

    # Construct and return the ExtractedData object based on the processor's output.
    # The DocumentProcessor's prompt is geared towards the LLMResponseModel which includes
    # a 'document_type' and 'structured_data' (which can be PSEGData, InvoiceData, etc.).
    # The 'doc_type' from processor.process_text() is the LLM's determination.
    # The 'structured_info' is the Pydantic model instance (e.g., PSEGData).

    if doc_type == "pseg_bill":
        # structured_info should ideally be PSEGData here
        # The LLMResponseModel and DocumentProcessor are designed to produce this.
        # We assume structured_info is a dict-like object or Pydantic model that can be passed to PSEGData
        # if it's not already a PSEGData instance.
        # For now, we pass it directly as it should be handled by the Pydantic union in LLMResponseModel.
        return ExtractedData(
            raw_text=ocr_text,
            document_type=doc_type,
            summary=f"Extracted PSEG bill data using model {model_name}.",
            structured_info=structured_info # This should be PSEGData model output
        )
    elif doc_type == "invoice":
        return ExtractedData(
            raw_text=ocr_text,
            document_type=doc_type,
            summary=f"Extracted invoice data using model {model_name}.",
            structured_info=structured_info # This should be InvoiceData model output
        )
    elif doc_type == "receipt":
        return ExtractedData(
            raw_text=ocr_text,
            document_type=doc_type,
            summary=f"Extracted receipt data using model {model_name}.",
            structured_info=structured_info # This should be ReceiptData model output
        )
    elif doc_type == "other":
         # structured_info here should be {"note": "description"}
        note = "Document classified as 'other'."
        if isinstance(structured_info, dict) and "note" in structured_info:
            note = structured_info["note"]
        elif hasattr(structured_info, 'note'): # If it's an object with a note attribute
            note = structured_info.note

        return ExtractedData(
            raw_text=ocr_text,
            document_type="other",
            summary=note,
            structured_info=structured_info # Store the original {"note": ...} or similar
        )
    else: # Should not happen if LLMResponseModel is adhered to
        return ExtractedData(
            raw_text=ocr_text,
            document_type="unknown_error",
            summary=f"Unknown document type '{doc_type}' returned by LLM.",
            structured_info={"error": f"Unknown document type '{doc_type}'"}
        )

# Example usage (for direct testing of this module, not part of the main app flow typically)
# Removing the __main__ block as per user request.

# Note: The original single-function approach process_document_with_llm has been deprecated
# in favor of the DocumentProcessor class and the more specific process_ocr_text_with_llm.
# The old function (if it existed here) would be removed.
# This file structure now favors explicit classification then specialized extraction if needed.

# A helper function for formatting model names, used by both classification and DocumentProcessor
# (This was refactored to be at the top of the file: _get_ollama_model_identifier) 