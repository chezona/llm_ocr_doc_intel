# Advanced Document Intelligence Pipeline with Docling, LLMs, and Conductor

## 1. Introduction and Goals

This project implements a robust, containerized pipeline for advanced intelligent document processing. It leverages:
*   **`LlamaParse` library:** For converting documents (e.g., PDFs) to Markdown format.
*   **Large Language Models (LLMs):** For extracting structured information, guided by `instructor`.
*   **Netflix Conductor:** For orchestrating the overall workflow, including a Human-in-the-Loop (HITL) step for LLM extraction failures.
*   **pip and `requirements.txt`:** For dependency management.

The workflow processes documents (initially PSEG utility bills) as follows:
1.  **Document Processing Task (`doc_processing_task`):**
    *   Parses the input document using `LlamaParse` (`app.llama_parser_processor.py`) to get Markdown.
    *   Attempts to extract structured data (e.g., `PSEGData` model) from the Markdown using an LLM (`app.llm_extractor.py`).
    *   Outputs processing status (`LLM_SUCCESS` or `LLM_FAILURE`) and relevant data for the next step.
2.  **LLM Quality Gate (DECISION Task):**
    *   If LLM extraction was successful, routes to direct database storage.
    *   If LLM extraction failed, routes to a HUMAN task for manual review.
3.  **Direct Database Storage Task (`store_document_data_task`):**
    *   Stores successfully LLM-extracted data in PostgreSQL via `app.db_utils.py`.
4.  **Human Review Task (`pseg_bill_human_review_task` - HUMAN Task):**
    *   Pauses the workflow, awaiting human input via the Conductor UI.
    *   The human reviewer inspects the OCR text (from `docling`) and provides corrected structured data as JSON.
5.  **Process Human Corrected Data Task (`process_human_corrected_data_task`):**
    *   Takes the human-provided JSON, validates it into `PSEGData`.
    *   Stores the corrected data in PostgreSQL via `app.db_utils.py`.

This architecture aims for:
*   **High Accuracy:** Combining automated extraction with a HITL fallback.
*   **Modularity & Extensibility:** Clear separation of concerns.
*   **Robustness:** Handles LLM failures gracefully through human intervention.
*   **Reliable Structured Output:** `instructor` and Pydantic models ensure data integrity.

The system is managed with Docker and Docker Compose.

## 2. Current Status & Key Capabilities

*   **Core Pipeline Functional:** The end-to-end pipeline including HITL for LLM failures is implemented.
*   **Initial Focus: PSEG Utility Bills:** System tuned for PSEG bills.
*   **`LlamaParse` Integration:** `LlamaParse` is used for PDF to Markdown conversion.
*   **`docling` Suspended:** The `docling` based parsing path is currently suspended to simplify dependencies. The code remains in the repository for potential future use.
*   **LLM Configuration:** Uses Ollama-served models, `instructor`, and `litellm`.
*   **HITL for LLM Extraction:** If `extract_pseg_data_from_markdown` fails (returns `None`), workflow routes to a HUMAN task in Conductor. Human provides corrected JSON via Conductor UI.
*   **Configuration:** Centralized in `app/config.py` (Pydantic `BaseSettings`).
*   **Dependency Management:** pip and `requirements.txt`.

### Document Parsing
The system now primarily uses **`LlamaParse`** for converting input files (e.g., PDFs) into Markdown suitable for LLM processing. The `docling` parser integration is currently suspended.

