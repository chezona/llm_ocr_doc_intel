"""
This module provides utilities for enhancing text, particularly OCR output, using an LLM.
It focuses on correcting OCR errors and can assess the quality of the correction.
"""
# import asyncio # Removed as it was only used for the local __main__ example.
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Optional, Tuple
import logging
import re
import time

from app.config_models import LLMConfig
from app.config import OLLAMA_BASE_URL # Import base URL

logger = logging.getLogger(__name__)

class LLMTextEnhancer:
    """
    Handles LLM-based text enhancement, focusing on OCR correction.
    It uses LangChain with an Ollama LLM, managing text chunking for large inputs
    and providing a method for quality assessment of the enhancements.
    """
    
    def __init__(self, llm_config: LLMConfig):
        """
        Initializes the LLMTextEnhancer with an LLM configuration.

        Sets up the Ollama LLM instance and Langchain processing chains for OCR
        correction and quality assessment based on the provided LLMConfig.
        """
        self.config = llm_config
        self.llm = Ollama(
            model=self.config.model_name,
            temperature=self.config.temperature,
            base_url=OLLAMA_BASE_URL # Use configured Ollama base URL
            # Consider adding timeout configurations from llm_config if available and supported by Langchain's Ollama
        )
        logger.info(f"Initialized LLMTextEnhancer with model: {self.config.model_name} at {OLLAMA_BASE_URL}")
        self._setup_chains()
        
    def _setup_chains(self):
        """
        Initializes and configures the LangChain LLMChains used by this class.

        This private helper method sets up chains for OCR correction and quality assessment,
        each with its specific prompt template tailored for the task.
        """
        # OCR Correction Chain
        ocr_correction_template = """Correct OCR-induced errors in the text, ensuring it flows coherently with the previous context. Follow these guidelines:

1. Fix OCR-induced typos and errors:
   - Correct words split across line breaks
   - Fix common OCR errors (e.g., 'rn' misread as 'm', 'I' vs 'l' vs '1')
   - Use context and common sense to correct errors
   - Only fix clear errors, do not alter the content unnecessarily
   - Do not add extra periods or any unnecessary punctuation unless grammatically required by the correction.

2. Maintain original structure:
   - Keep all headings and subheadings intact.
   - Preserve paragraph breaks where they clearly delineate distinct blocks of text.

3. Preserve original content:
   - Keep all important information from the original text (names, dates, numbers, addresses, specific terms).
   - Do not add any new information not present in the original text.
   - Remove unnecessary line breaks within sentences or paragraphs that fragment continuous text.
   
4. Maintain coherence:
   - Ensure the content connects smoothly with the previous context.
   - Handle text that starts or ends mid-sentence appropriately.

Previous context (last 500 characters, if any):
{prev_context}

Current chunk to process:
{chunk_text}

Corrected text (output ONLY the corrected version of the 'Current chunk to process'):
"""
        
        ocr_prompt = PromptTemplate(
            template=ocr_correction_template,
            input_variables=["prev_context", "chunk_text"]
        )
        
        self.ocr_correction_chain = LLMChain(
            llm=self.llm, 
            prompt=ocr_prompt
        )
        
        # Markdown Formatting Chain - REMOVED
        
        # Quality assessment chain
        quality_template = """Compare the following samples of original OCR text with the processed output and assess the quality of the processing. Consider the following factors:
1. Accuracy of error correction (e.g., typos fixed, garbled text clarified).
2. Improvement in readability and coherence.
3. Preservation of original content, meaning, and essential details (names, dates, numbers).
4. Avoidance of introducing new information or hallucinations.
5. Proper handling of structure (paragraphs, headings if present).

Original text sample:
```
{original_text}
```

Processed text sample:
```
{processed_text}
```

Provide a quality score between 0 and 100, where 100 is perfect processing. Also provide a brief explanation of your assessment.
Your response should be in the following format:
SCORE: [Your score]
EXPLANATION: [Your explanation]
"""
        
        quality_prompt = PromptTemplate(
            template=quality_template,
            input_variables=["original_text", "processed_text"]
        )
        
        self.quality_chain = LLMChain(
            llm=self.llm, 
            prompt=quality_prompt
        )
        
    def create_text_chunks(self, full_text: str, chunk_size: int = 8000, overlap: int = 200) -> List[str]: # Increased overlap slightly
        """
        Splits a large text string into smaller, manageable chunks for LLM processing.

        Uses LangChain's RecursiveCharacterTextSplitter with a specified chunk size
        and overlap to maintain context between chunks, returning a list of text strings.
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap, # Ensure overlap is less than chunk_size
            length_function=len,
            is_separator_regex=False,
            separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""] # Common separators
        )
        return text_splitter.split_text(full_text)
    
    async def _process_single_chunk_for_correction(self, 
                           chunk: str, 
                                               prev_context: str) -> str:
        """
        Processes a single text chunk for OCR correction.

        Args:
            chunk: The text chunk to process.
            prev_context: The context from the previous chunk.

        Returns:
            The corrected text for the current chunk.
        """
        corrected_text_raw = await self.ocr_correction_chain.arun(
            prev_context=prev_context[-500:], # Use last 500 chars of prev context
            chunk_text=chunk
        )
        # It's crucial that the LLM is prompted to ONLY return the corrected chunk,
        # otherwise, self.remove_corrected_text_header might be needed or more complex parsing.
        # Assuming the new prompt instruction "Corrected text (output ONLY the corrected version of the 'Current chunk to process'):" works.
        return corrected_text_raw.strip() 

    async def enhance_text(self, full_text: str) -> str:
        """
        Enhances the full input text by applying OCR correction chunk by chunk.

        Args:
            full_text: The entire raw OCR text to enhance.

        Returns:
            The fully enhanced text after OCR correction.
        """
        if not full_text or full_text.isspace():
            logger.warning("Input text for enhancement is empty or whitespace.")
            return ""

        # Determine appropriate chunk_size based on model, e.g. from self.config if available
        # For now, using a default that's generally safe.
        # Max context window for Llama2 models is often 4096 tokens. 8000 chars might be too large.
        # Let's use a smaller chunk size, e.g., 2000 chars, assuming ~4 chars/token.
        # This should be configurable or derived from model metadata if possible.
        chunk_size = self.config.chunk_size if hasattr(self.config, 'chunk_size') and self.config.chunk_size else 2000
        chunk_overlap = self.config.chunk_overlap if hasattr(self.config, 'chunk_overlap') and self.config.chunk_overlap else 200
        
        if chunk_overlap >= chunk_size:
            chunk_overlap = int(chunk_size / 4) # Ensure overlap is smaller than chunk_size
            logger.warning(f"Chunk overlap was >= chunk size. Adjusted overlap to {chunk_overlap}")


        chunks = self.create_text_chunks(full_text, chunk_size=chunk_size, overlap=chunk_overlap)
        
        if not chunks:
            logger.warning("Text splitting resulted in no chunks.")
            return ""

        processed_chunks = []
        current_prev_context = ""
        
        logger.info(f"Starting OCR correction for {len(chunks)} chunks.")

        for i, chunk_text in enumerate(chunks):
            logger.info(f"Processing chunk {i + 1}/{len(chunks)} (length: {len(chunk_text)} characters)")
            start_time = time.time()
            
            corrected_chunk = await self._process_single_chunk_for_correction(
                chunk=chunk_text,
                prev_context=current_prev_context
            )
            processed_chunks.append(corrected_chunk)
            current_prev_context = corrected_chunk # The corrected chunk becomes context for the next
            
            # Correctly indented lines:
            processing_time = time.time() - start_time
            logger.info(f"Chunk {i + 1}/{len(chunks)} corrected in {processing_time:.2f}s. Output length: {len(corrected_chunk)} characters")
        # End of for loop
        return "\n\n".join(processed_chunks) # Join with double newline to somewhat preserve paragraphing if chunks align
        
    async def assess_text_quality(self, original_text: str, processed_text: str) -> Tuple[Optional[int], Optional[str]]: # Renamed for clarity
        """
        Assesses the quality of processed text compared to the original OCR text via an LLM.

        It sends samples of original and processed text to an LLM to get a quality score
        (0-100) and a textual explanation. Returns (score, explanation) or (None, None) on error.
        Samples are taken from the beginning of the text.
        """
        if not original_text or not processed_text:
            logger.warning("Cannot assess quality with empty original or processed text.")
            return None, None

        # Limit text lengths to avoid token limits, focusing on the start of the document
        # Max chars should be less than model's context window, considering prompt and output.
        # A 7500 char sample for original and 7500 for processed is very large for many models.
        # Let's reduce this significantly. E.g. 2000 chars each.
        max_chars_per_sample = self.config.quality_assessment_sample_chars if hasattr(self.config, 'quality_assessment_sample_chars') and self.config.quality_assessment_sample_chars else 2000
        
        original_sample = original_text[:max_chars_per_sample]
        processed_sample = processed_text[:max_chars_per_sample]
        
        logger.info(f"Assessing quality. Original sample length: {len(original_sample)}, Processed sample length: {len(processed_sample)}")
        
        try:
            response = await self.quality_chain.arun( # Changed to arun for async
                original_text=original_sample,
                processed_text=processed_sample
            )
            
            score_match = re.search(r"SCORE:\s*(\d+)", response, re.IGNORECASE)
            explanation_match = re.search(r"EXPLANATION:\s*(.+)", response, re.IGNORECASE | re.DOTALL)
            
            score = int(score_match.group(1)) if score_match else None
            explanation = explanation_match.group(1).strip() if explanation_match else None
            
            if score is None:
                logger.warning(f"Could not parse quality score from LLM response. Response: '{response[:200]}...'")
            if explanation is None:
                 logger.warning(f"Could not parse quality explanation from LLM response. Response: '{response[:200]}...'")

            logger.info(f"Quality assessment result - Score: {score}, Explanation: {explanation[:100] if explanation else 'N/A'}...")
            return score, explanation
        
        except Exception as e:
            logger.error(f"Error during quality assessment: {e}", exc_info=True)
            return None, None

    # Removed remove_corrected_text_header and remove_markdown_header.
    # The OCR correction prompt now explicitly asks the LLM to only output the corrected chunk.
    # If issues persist with LLM adding extra headers, these might need to be reinstated or refined.

async def enhance_ocr_text(raw_text: str, llm_config: LLMConfig) -> str:
        """
    Top-level function to enhance raw OCR text using the LLMTextEnhancer.

    Instantiates LLMTextEnhancer and calls its enhance_text method.

    Args:
        raw_text: The raw OCR text to enhance.
        llm_config: Configuration for the LLM to be used for enhancement.

    Returns:
        The enhanced text.
        """
        if not raw_text:
            return ""
        try:
            enhancer = LLMTextEnhancer(llm_config=llm_config)
            enhanced_text = await enhancer.enhance_text(full_text=raw_text)
            return enhanced_text
        except Exception as e:
            logger.error(f"Error in enhance_ocr_text: {e}", exc_info=True)
            return raw_text # Fallback to raw text on error

# Removed the if __name__ == '__main__': block and main_example function
# as per user request to eliminate local testing code.
# The asyncio import at the top might be solely for this example, if so it could be removed too,
# but will leave it for now unless it causes issues or is clearly unused elsewhere.
# The logging import is likely used by the logger instance in the class, so it should stay. 