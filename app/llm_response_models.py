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
import datetime

class PSEGLineItem(BaseModel):
    description: Optional[str] = Field(None, description="Description of the line item charge")
    amount: Optional[float] = Field(None, description="Amount of the line item charge")

class PSEGData(BaseModel):
    account_number: Optional[str] = Field(None, description="PSEG account number")
    customer_name: Optional[str] = Field(None, description="Customer name")
    service_address: Optional[str] = Field(None, description="Address where service is provided")
    billing_address: Optional[str] = Field(None, description="Customer billing address")
    billing_date: Optional[datetime.date] = Field(None, description="Date the bill was issued (bill_date)")
    billing_period_start_date: Optional[datetime.date] = Field(None, description="Start date of the billing period")
    billing_period_end_date: Optional[datetime.date] = Field(None, description="End date of the billing period")
    due_date: Optional[datetime.date] = Field(None, description="Date the payment is due")
    total_amount_due: Optional[float] = Field(None, description="Total amount due on the bill")
    previous_balance: Optional[float] = Field(None, description="Previous balance amount")
    payments_received: Optional[float] = Field(None, description="Total payments received")
    current_charges: Optional[float] = Field(None, description="Total current charges")
    line_items: Optional[List[PSEGLineItem]] = Field(None, description="List of line item charges")
    raw_text_summary: Optional[str] = Field(None, description="A brief summary of OCRed text for context, if needed")

    class Config:
        extra = 'ignore' # Allow model to ignore extra fields from LLM output if any

# Example of another document type model (can be expanded later)
class GenericInvoiceData(BaseModel):
    invoice_number: Optional[str] = None
    vendor_name: Optional[str] = None
    invoice_date: Optional[datetime.date] = None
    total_amount: Optional[float] = None
    raw_text: Optional[str] = None 