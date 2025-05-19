"""
Defines Pydantic models for holding OCR and LLM configuration settings.

These models provide a structured and validated way to pass around complex 
configuration parameters to functions and classes that perform OCR and LLM operations.
Using Pydantic models for configuration offers several advantages:
- **Type Safety:** Ensures that configuration parameters are of the correct type.
- **Validation:** Allows for defining validation rules (e.g., min/max values) for parameters.
- **Clear Defaults:** Default values are explicitly defined within the models.
- **Readability & Maintainability:** Makes the configuration explicit and easier to 
  understand and modify compared to passing around many individual arguments or 
  unstructured dictionaries.
- **Reusability:** These configuration objects can be easily instantiated and reused 
  across different parts of the application that require similar settings.

For example, `OCRConfig` can be passed to an OCR processing function to control 
page limits and formatting, while `LLMConfig` can be passed to an LLM interaction 
function to specify the model, temperature, and token limits.
"""
from pydantic import BaseModel, Field
from typing import Optional

class OCRConfig(BaseModel):
    """
    Configuration settings for Optical Character Recognition (OCR) processing.
    
    This model encapsulates parameters that control the behavior of the OCR engine 
    when processing documents (e.g., PDFs or images). It allows for consistent 
    application of OCR settings across different parts of the system that might 
    invoke OCR. The defaults are set to common general-purpose values.
    """
    max_pages: int = Field(
        0, 
        description="Maximum number of pages to process from the input document. A value of 0 (default) means all pages will be processed. This is useful for limiting processing time or resources for very long documents."
    )
    skip_first_n_pages: int = Field(
        0, 
        description="Number of pages to skip from the beginning of the document before starting OCR. Default is 0. This can be used to ignore cover pages or irrelevant introductory material."
    )
    reformat_as_markdown: bool = Field(
        True, 
        description="If True (default), attempts to reformat the raw OCR text into a basic Markdown structure. This can sometimes improve the readability and structure of the text for downstream LLM processing, helping the LLM better understand layout elements like headings and paragraphs."
    )
    suppress_headers_and_page_numbers: bool = Field(
        True, 
        description="If True (default), attempts to identify and remove repeating headers, footers, and page numbers from the OCR output. This is generally desirable as such elements are often noise for LLM data extraction tasks."
    )
    
class LLMConfig(BaseModel):
    """
    Configuration settings for Large Language Model (LLM) interactions.
    
    This model groups parameters that control the behavior of LLMs during text 
    processing or generation tasks, such as text enhancement or data extraction.
    It provides a way to standardize LLM call parameters. The defaults are chosen 
    to be generally applicable, but can be overridden for specific tasks or models.
    """
    model_name: str = Field(
        "gemma:7b", # Note: This default might differ from global config in app.config.py; should be used judiciously.
        description="The specific LLM model name to be used for a particular task (e.g., text enhancement, classification, extraction). This allows different tasks to potentially use different models best suited for them. It might override a global model setting from `app.config` for a specific operation."
    )
    temperature: float = Field(
        0.7, 
        description="Controls the randomness of the LLM's output. Lower values (e.g., 0.2) make the output more deterministic and focused, while higher values (e.g., 0.8) make it more creative and diverse. 0.7 is a common default for a balance.",
        ge=0.0, 
        le=2.0
    )
    max_tokens: int = Field(
        2048, 
        description="The maximum number of tokens (words/sub-words) that the LLM is allowed to generate in its response. This helps control the length of the output and prevent runaway generation. The chosen value should be appropriate for the expected output length and the model's context window."
    )
    token_buffer: int = Field(
        500, 
        description="A buffer to subtract from the model's theoretical maximum context window when calculating available space for input. This accounts for potential inaccuracies in token counting or space reserved by the model for its own use."
    )
    token_cushion: int = Field(
        300, 
        description="An additional safety margin subtracted from available tokens. The idea is to not use the absolute maximum available tokens for the input prompt to avoid unexpectedly hitting context limits, especially if the prompt length estimation is not perfectly accurate."
    )
    
    # --- Fields primarily for LLMTextEnhancer or similar chunk-based text processing tasks ---
    # These parameters are particularly relevant when processing large texts that need to be 
    # broken down into smaller chunks before being fed to an LLM, due to context window limitations.
    chunk_size: Optional[int] = Field(
        None, # Default to None, should be set based on model context window & task.
        description="The target size (e.g., in characters or tokens) for each chunk of text when splitting a large document for LLM processing. Effective chunking is key to processing long documents."
    )
    chunk_overlap: Optional[int] = Field(
        None, # Default to None, typically 10-20% of chunk_size if used.
        description="The number of characters or tokens that will overlap between consecutive chunks. Overlap helps maintain context and continuity when processing text in segments, reducing the chance of information being lost at chunk boundaries."
    )
    quality_assessment_sample_chars: Optional[int] = Field(
        None, # Default to None, e.g., 500-1000 if used.
        description="Number of characters from the beginning and end of processed text to use for a qualitative assessment (e.g., by another LLM call) of the enhancement or processing quality. Used in `LLMTextEnhancer`."
    )

    # --- Potential future LLM parameters ---
    # The commented-out fields below are examples of other common LLM parameters that 
    # could be added to this model if more fine-grained control over LLM generation 
    # is needed for specific tasks.
    # top_p: Optional[float] = Field(None, description="Top-p (nucleus) sampling probability.", ge=0.0, le=1.0)
    # top_k: Optional[int] = Field(None, description="Top-k sampling: number of highest probability vocabulary tokens to keep.", ge=0)
    # base_url: Optional[str] = Field(None, description="Base URL for a specific LLM API, if this config needs to override a global one.")
    # api_key: Optional[str] = Field(None, description="API key for a specific LLM service, if required and not handled globally.") 