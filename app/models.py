"""
Defines Pydantic models used for structuring data within the application, 
particularly for data extracted from documents after OCR and LLM processing.

These models serve several key purposes:
1.  **Data Validation:** Pydantic enforces type hints at runtime, ensuring that data 
    conforms to the expected structure and types. This is crucial when dealing with 
    potentially noisy or variable output from LLMs.
2.  **Clear Data Contracts:** They define a clear schema for what data to expect, 
    making it easier for different parts of the application (e.g., LLM extraction, 
    database storage, API responses) to interoperate reliably.
3.  **Developer Experience:** Auto-completion, type checking, and clear error messages 
    from Pydantic improve the development process.
4.  **Serialization/Deserialization:** Pydantic models can be easily serialized to JSON 
    (e.g., for API outputs or storing in NoSQL DBs if needed) and deserialized from 
    Python dictionaries.

This module includes a detailed model for PSEG utility bills (`PSEGData`) and a 
general container (`ExtractedData`) for holding OCR results alongside structured information.
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Union

class PSEGData(BaseModel):
    """
    Represents structured data specific to a PSEG utility bill.
    
    This model is designed to capture all relevant pieces of information typically found 
    on a PSEG bill. Each field is defined as `Optional` because the LLM extraction 
    process might not always find every piece of information, or some information might 
    genuinely be missing from certain bills. Using `Optional` provides flexibility and 
    avoids validation errors if a field isn't populated.
    The `Field(description=...)` provides metadata that can be used by `instructor` 
    when generating prompts for the LLM, helping the LLM understand what kind of 
    information is expected for each field.
    """
    account_number: Optional[str] = Field(None, description="The unique PSEG account identifier.")
    bill_date: Optional[str] = Field(None, description="The date the bill was issued by PSEG, preferably in YYYY-MM-DD format.")
    due_date: Optional[str] = Field(None, description="The date by which the payment for the bill is due, preferably in YYYY-MM-DD format.")
    service_address: Optional[str] = Field(None, description="The full street address where the utility services (gas/electric) were provided.")
    billing_address: Optional[str] = Field(None, description="The mailing address for the bill, if it is different from the service address.")
    customer_name: Optional[str] = Field(None, description="The full name of the PSEG account holder as it appears on the bill.")
    billing_period_start: Optional[str] = Field(None, description="The start date of the billing cycle covered by this bill, preferably in YYYY-MM-DD format.")
    billing_period_end: Optional[str] = Field(None, description="The end date of the billing cycle covered by this bill, preferably in YYYY-MM-DD format.")
    total_amount_due: Optional[float] = Field(None, description="The total monetary amount that is due for payment on this bill.")
    previous_balance: Optional[float] = Field(None, description="The outstanding balance amount from the previous billing period, if any.")
    payments_received: Optional[float] = Field(None, description="The total amount of payments received by PSEG since the last bill was issued.")
    current_charges: Optional[float] = Field(None, description="The sum of all new charges incurred during the current billing period.")
    next_meter_reading: Optional[str] = Field(None, description="The scheduled date for the next meter reading, if mentioned, preferably in YYYY-MM-DD format.")
    utility_company: Optional[str] = Field(None, description="The name of the utility company, typically 'PSE&G' or 'Public Service Electric and Gas Company'.")
    customer_service_phone: Optional[str] = Field(None, description="The primary customer service phone number provided on the bill.")
    payment_methods: Optional[str] = Field(None, description="A brief description or list of available payment options mentioned on the bill (e.g., 'Online, By Mail, By Phone').")
    
    # Commented out fields below are examples of further details that could be added 
    # if deeper extraction of usage or specific line items is required in the future.
    # gas_usage_ccf: Optional[float] = Field(None, description="Gas usage in CCF (Hundred Cubic Feet).")
    # electric_usage_kwh: Optional[float] = Field(None, description="Electric usage in kWh (kilowatt-hours).")
    # service_period_start: Optional[str] = Field(None, description="Start date of the service period if different from billing period, e.g., YYYY-MM-DD")
    # service_period_end: Optional[str] = Field(None, description="End date of the service period if different from billing period, e.g., YYYY-MM-DD")

class ExtractedData(BaseModel):
    """
    A general-purpose container model for the output of the document processing pipeline.

    This model is designed to be the standard output structure from the LangGraph pipeline
    (`langgraph_pipeline_task`) and serves as the input to the database storage task 
    (`db_storage_task`). It encapsulates all key pieces of information derived from a document:
    - The raw text from OCR.
    - The classified document type (which determines subsequent handling).
    - The structured data extracted (e.g., `PSEGData` if it's a PSEG bill).
    - An optional summary, useful if detailed structured extraction isn't applicable or fails.
    
    This structure ensures a consistent data flow between different stages of the workflow.
    """
    raw_text: str = Field(description="The complete raw text extracted from the document by the OCR process. This is preserved for auditing and potential re-processing.")
    document_type: Optional[str] = Field(None, description="The type of document identified by the initial classification step (e.g., 'pseg_bill', 'invoice', 'receipt', 'other'). This dictates further processing logic.")
    structured_info: Optional[PSEGData] = Field(None, description="If the document is identified as a PSEG bill and extraction is successful, this field will contain the populated PSEGData model. For other document types, it may be None or contain a more generic structure if implemented.")
    summary: Optional[str] = Field(None, description="A brief textual summary of the document's content. This can be generated by an LLM, especially for documents where detailed structured extraction is not performed or if the primary extraction fails, providing a fallback human-readable overview.") 