"""
Defines Pydantic models specifically for structuring the responses expected from 
Large Language Models (LLMs) when using the `instructor` library.

These models are crucial for achieving reliable and structured data extraction from LLMs.
`instructor` works by patching an LLM client (like OpenAI's or LiteLLM's) to 
coerce its output into a specified Pydantic model. This involves dynamically 
generating a JSON schema from the Pydantic model, adding it to the LLM prompt 
(often in system messages or function/tool calling definitions), and then parsing 
the LLM's JSON output back into an instance of the Pydantic model, including validation.

Key benefits of this approach:
- **Structured Output:** Forces the LLM to provide data in a predictable, usable format.
- **Validation:** Ensures the LLM's output adheres to the defined types and constraints.
- **Retries (with `instructor`):** `instructor` can automatically retry LLM calls if 
  parsing or validation fails, potentially with different parameters or by providing 
  error feedback to the LLM.
- **Clear Prompts:** The Pydantic model, especially field descriptions, helps in crafting 
  clear instructions for the LLM on what data to extract and how to format it.
"""
from pydantic import BaseModel, Field, RootModel
from typing import Union, Dict, Any, Optional, List

from app.models import PSEGData # Importing the detailed PSEGData model

class SimpleClassificationOutput(BaseModel):
    """
    A Pydantic model representing the output of a simple document classification task.
    
    This model is used by an LLM (via `instructor`) to return the identified type 
    of a document and an optional confidence score. It's designed for the initial, 
    coarse-grained classification step in the 2-pass LangGraph architecture.
    The primary purpose is to quickly determine if a document warrants further, more 
    specialized processing (e.g., if it's a PSEG bill).
    """
    document_type: str = Field(
        ..., 
        description="The single most likely classified document type (e.g., 'pseg_bill', 'invoice', 'receipt', 'bank_statement', 'other')."
    )
    confidence: Optional[float] = Field(
        None, 
        description="An optional confidence score for the classification, ranging from 0.0 (low confidence) to 1.0 (high confidence).", 
        ge=0.0, 
        le=1.0
    )
    # Rationale for using a simple string for document_type:
    # While an Enum could be used for predefined types, a string offers flexibility
    # if the LLM identifies types not explicitly in a predefined list, or if new 
    # types are to be supported without code changes to an Enum. Validation can occur downstream.

class LLMResponseModel(BaseModel):
    """
    A Pydantic model that defines the expected structured output from the LLM 
    for more detailed data extraction, specifically after a document has been 
    classified (e.g., as a PSEG bill).

    `instructor` uses this model to parse and validate the LLM's JSON response. 
    The model is designed to be flexible: if the document is a PSEG bill, 
    `structured_data` should conform to the `PSEGData` model. If it's classified 
    as 'other' or a type for which no detailed model exists, `structured_data` 
    can be a simple dictionary (e.g., containing a note).

    This model structure supports the conditional logic in the 2-pass architecture: 
    after classification, the system expects either detailed PSEG data or a 
    more generic output for other document types.
    """
    document_type: str = Field(
        ..., 
        description="The confirmed or re-classified document type after OCR and initial processing (e.g., 'pseg_bill', 'other'). This might reiterate or refine the type from SimpleClassificationOutput."
    )
    # The Union allows for type-specific structured data. For PSEG bills, it expects PSEGData.
    # For 'other' document types, or if PSEG extraction fails but classification was 'pseg_bill',
    # it can fall back to a dictionary, which might contain a note or partially extracted fields.
    # This provides a flexible way to handle varied extraction outcomes.
    structured_data: Union[PSEGData, Dict[str, Any]] = Field(
        ..., 
        description="The structured data extracted from the document. If document_type is 'pseg_bill', this should be PSEGData. For 'other' types, it can be a dictionary, possibly with a 'note' field explaining why detailed extraction was not performed or was not applicable."
    )

    class Config:
        """
        Pydantic model configuration.
        The `json_schema_extra` with examples can be very useful for `instructor` 
        as it can pass these examples to the LLM as part of the prompt, 
        guiding it towards the desired output format (few-shot prompting).
        This is commented out as it's an optional feature and requires careful crafting
        of examples that align with the LLM's capabilities and the prompt.
        """
        # json_schema_extra = {
        #     "examples": [
        #         {
        #             "document_type": "pseg_bill",
        #             "structured_data": {
        #                 # Example fields from PSEGData
        #                 "account_number": "1234567890",
        #                 "bill_date": "2023-10-26",
        #                 "due_date": "2023-11-15",
        #                 "service_address": "123 Main St, Anytown, NJ 07001",
        #                 "total_amount_due": 150.75,
        #                 "customer_name": "John Doe"
        #             }
        #         },
        #         {
        #             "document_type": "other_invoice",
        #             "structured_data": {
        #                 "invoice_id": "INV-2023-001",
        #                 "vendor_name": "Some Company LLC",
        #                 "total_amount": 75.20,
        #                 "note": "Generic invoice processed."
        #             }
        #         },
        #         {
        #             "document_type": "other",
        #             "structured_data": {
        #                 "note": "Document content is unclear or not a supported type for detailed extraction."
        #             }
        #         }
        #     ]
        # }
        pass 