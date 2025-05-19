"""
Defines Pydantic models for structuring data extracted from documents.
This includes models for PSEG utility bills, and a container
model to hold the raw OCR text along with the extracted structured information.
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Union

class PSEGData(BaseModel):
    """
    Represents structured data specific to a PSEG utility bill.
    Fields include account number, bill dates, service address, and various charge amounts.
    All fields are optional.
    """
    account_number: Optional[str] = Field(None, description="PSEG account number")
    bill_date: Optional[str] = Field(None, description="Date the bill was issued, e.g., YYYY-MM-DD")
    due_date: Optional[str] = Field(None, description="Payment due date, e.g., YYYY-MM-DD")
    service_address: Optional[str] = Field(None, description="Address where services were provided")
    billing_address: Optional[str] = Field(None, description="Mailing address if different from service address")
    customer_name: Optional[str] = Field(None, description="Full name of the account holder")
    billing_period_start: Optional[str] = Field(None, description="Start date of billing cycle, e.g., YYYY-MM-DD")
    billing_period_end: Optional[str] = Field(None, description="End date of billing cycle, e.g., YYYY-MM-DD")
    total_amount_due: Optional[float] = Field(None, description="Total amount due on the bill")
    previous_balance: Optional[float] = Field(None, description="Previous balance, if any")
    payments_received: Optional[float] = Field(None, description="Payments received since last bill")
    current_charges: Optional[float] = Field(None, description="Total current month's charges")
    next_meter_reading: Optional[str] = Field(None, description="Date for the next scheduled meter reading, e.g., YYYY-MM-DD")
    utility_company: Optional[str] = Field(None, description="Name of the utility company, e.g., PSE&G")
    customer_service_phone: Optional[str] = Field(None, description="Customer service phone number")
    payment_methods: Optional[str] = Field(None, description="Brief description or list of available payment options mentioned")
    # Add more PSEG-specific fields if needed, e.g., usage details
    # gas_usage_ccf: Optional[float] = Field(None, description="Gas usage in CCF")
    # electric_usage_kwh: Optional[float] = Field(None, description="Electric usage in kWh")
    # service_period_start: Optional[str] = Field(None, description="Start date of the service period")
    # service_period_end: Optional[str] = Field(None, description="End date of the service period")

class ExtractedData(BaseModel):
    """
    A container model for the overall result of OCR and LLM processing.
    It holds the raw OCR text, the identified document type (e.g., 'pseg_bill', 'other'),
    the extracted structured data (PSEGData if applicable), and an optional summary.
    """
    raw_text: str = Field(description="The complete raw text extracted from the OCR process.")
    # Union simplified to Optional[PSEGData]
    structured_info: Optional[PSEGData] = Field(None, description="Structured information extracted from the text (PSEGData if applicable).")
    document_type: Optional[str] = Field(None, description="Type of document identified (e.g., 'pseg_bill', 'other')")
    summary: Optional[str] = Field(None, description="A brief summary of the document content if structured extraction is not applicable or fails.") 