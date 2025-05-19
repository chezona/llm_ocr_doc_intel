from pydantic import BaseModel, Field
from typing import Union, Dict, Any, Optional
from app.models import PSEGData

class SimpleClassificationOutput(BaseModel):
    """
    Defines a simple output model for document classification only.
    """
    document_type: str = Field(..., description="The classified document type (e.g., 'pseg_bill', 'invoice', 'receipt', 'other').")
    confidence: Optional[float] = Field(None, description="Optional confidence score for the classification (0.0 to 1.0).", ge=0, le=1)

class LLMResponseModel(BaseModel):
    """
    Defines the expected structured output from the LLM after processing.
    Instructor will use this model to parse and validate the LLM's response.
    Now focused on PSEG bills or 'other' document types.
    """
    document_type: str = Field(..., description="The classified document type (e.g., 'pseg_bill', 'other').")
    structured_data: Union[PSEGData, Dict[str, Any]] = Field(..., description="The structured data extracted, corresponding to the document_type. For 'other', this can be an empty dict or a dict with a 'note'.")

    class Config:
        # Example of how to add examples for the model, useful for some LLMs/instructor features
        # Updated examples to reflect PSEG-only focus
        # json_schema_extra = {
        #     "examples": [
        #         {
        #             "document_type": "pseg_bill",
        #             "structured_data": {
        #                 "account_number": "1234567890",
        #                 "bill_date": "2023-10-26",
        #                 "total_amount_due": 150.75
        #             }
        #         },
        #         {
        #             "document_type": "other",
        #             "structured_data": {"note": "Document content is unclear or not supported."}
        #         }
        #     ]
        # }
        pass 