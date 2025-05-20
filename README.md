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

## 3. System Architecture

The system uses Conductor to orchestrate tasks, including a decision point for routing to human review if LLM extraction fails.

### 3.1. Architectural Diagram

```mermaid
graph TD
    UserInput[/\"User Input <br/> (PDF, Image, DOCX etc. via cURL)\"/] --> CW_API{{"""Conductor API <br/> (POST /api/workflow/document_processing_workflow)"""}};

    subgraph ConductorSystem["Conductor Server & UI"]
        CW_API --> CS["Conductor Server"];
        CS <--> CUI["Conductor UI"];
        CS --> Workflow["document_processing_workflow <br/> (ocr_processing_workflow.json, v2)"];
    end

    Workflow -- "Schedules doc_processing_task" --> AppWorkerService["App Worker Service <br/> (app_worker container)"];

    subgraph AppWorkerServiceContext ["App Worker: doc_processing_task Execution"]
        direction LR
        AppWorkerService -.-> DPTask["doc_processing_task <br/> (app.conductor_workers.process_document_task)"];
        DPTask -- "Input: document_path, <br/> document_type_hint" --> ParserLogic["Document Parser <br/> (LlamaParse)"];
        ParserLogic -- "Output: Markdown Text" --> LLMExtractorWrapper["LLM Extraction Logic <br/> (app.llm_extractor)"];
        LLMExtractorWrapper -- "Output: Optional[PSEGData]" --> DPTask;
        DPTask -- "Output: <br/> status_doc_processing, <br/> db_storage_payload OR human_review_input" --> Workflow;
    end

    Workflow -- "Routes based on status_doc_processing" --> LLMQualityGate{{"llm_quality_gate_decision <br/> (DECISION Task)"}};

    LLMQualityGate -- "LLM_SUCCESS" --> DBStoreTask["store_document_data_task <br/> (app.conductor_workers.store_document_data_task)"];
    subgraph AppWorkerDBDirectContext ["App Worker: store_document_data_task Execution"]
        direction LR
        AppWorkerService -.-> DBStoreTask;
        DBStoreTask -- "Input: db_storage_payload" --> DBUtilsDirect["app.db_utils.insert_pseg_data"];
        DBUtilsDirect -- "Stores data" --> PGSQLServiceDB["PostgreSQL DB"];
        DBStoreTask -- "Output: db_direct_output" --> Workflow;
    end

    LLMQualityGate -- "LLM_FAILURE" --> HumanReviewTask["pseg_bill_human_review_task <br/> (HUMAN Task)"];
    subgraph HumanReviewContext ["Human Review via Conductor UI/API"]
        HumanReviewTask -- "Input: human_review_input <br/> (markdown_text, original_doc_path, etc.)" --> Human["Human Reviewer <br/> (Uses Conductor UI)"];
        Human -- "Provides output JSON: <br/> { corrected_pseg_data_dict: { ... } }" --> HumanReviewTask;
        HumanReviewTask -- "Output: corrected_pseg_data_dict" --> Workflow;
    end
    
    Workflow -- "Schedules process_human_corrected_data_task" --> AppWorkerService;
    subgraph AppWorkerDBHumanContext ["App Worker: process_human_corrected_data_task Execution"]
        direction LR
        AppWorkerService -.-> ProcessHumanDataTask["process_human_corrected_data_task <br/> (app.conductor_workers.process_human_corrected_data_task)"];
        ProcessHumanDataTask -- "Input: corrected_pseg_data_dict, <br/> markdown_text, document_type_hint" --> DBUtilsHuman["app.db_utils.insert_pseg_data"];
        DBUtilsHuman -- "Stores data" --> PGSQLServiceDB;
        ProcessHumanDataTask -- "Output: db_human_corrected_output" --> Workflow;
    end

    LLMExtractorWrapper -- "Uses" --> LiteLLMLib["LiteLLM Library"];
    LiteLLMLib -- "Communicates with" --> OllamaService["Ollama Service (LLMs)"];
    LLMExtractorWrapper -- "Uses" --> InstructorLib["Instructor Library"];
    LLMExtractorWrapper -- "Uses Pydantic Model" --> PSEGDataModel["PSEGData <br/> (app.llm_response_models)"];
    
    subgraph SettingsAndConfigRef ["Shared Configuration"]
      PydanticSettings["app.config.settings"]
    end
    DPTask -- "Reads config (implicitly for LlamaParse API key)" --> PydanticSettings;
    LLMExtractorWrapper -- "Reads config" --> PydanticSettings;
    DBUtilsDirect -- "Reads config (for DB URL)" --> PydanticSettings;
    DBUtilsHuman -- "Reads config (for DB URL)" --> PydanticSettings;

    classDef default fill:#fff,stroke:#333,stroke-width:2px,color:#333;
    classDef userInput fill:#f9f,stroke:#333,stroke-width:2px;
    classDef conductor fill:#lightblue,stroke:#333,stroke-width:2px;
    classDef appWorker fill:#lightgreen,stroke:#333,stroke-width:2px;
    classDef services fill:#orange,stroke:#333,stroke-width:2px;
    classDef database fill:#cyan,stroke:#333,stroke-width:2px;
    classDef extLib fill:#mediumpurple,stroke:#333,stroke-width:1.5px,color:white;
    classDef pydantic fill:#ffcc99,stroke:#333,stroke-width:1.5px;
    classDef human fill:#ffebcc,stroke:#333,stroke-width:2px;

    class UserInput userInput;
    class CW_API,CS,CUI,Workflow,LLMQualityGate,HumanReviewTask conductor;
    class AppWorkerService,DPTask,DBStoreTask,ProcessHumanDataTask appWorker;
    class PGSQLServiceDB database;
    class OllamaService services;
    class LiteLLMLib,InstructorLib extLib; /* Removed DoclingLib */
    class PSEGDataModel,PydanticSettings pydantic;
    class Human human;
```

### 3.2. Component Breakdown

*   **User Input:** Documents submitted via cURL.
*   **Netflix Conductor:** Orchestrates `document_processing_workflow` (v2) from `ocr_processing_workflow.json`.
    *   **`doc_processing_task`**:
        *   Calls `LlamaParse` and then LLM extraction.
        *   Outputs `status_doc_processing` ("LLM_SUCCESS" or "LLM_FAILURE").
        *   If "LLM_SUCCESS", outputs `db_storage_payload`.
        *   If "LLM_FAILURE", outputs `human_review_input` (containing `markdown_text`, `original_document_path`, `document_type_hint`).
    *   **`llm_quality_gate_decision` (DECISION Task):**
        *   Routes to `store_document_data_task` if `status_doc_processing` is "LLM_SUCCESS".
        *   Routes to `pseg_bill_human_review_task` if `status_doc_processing` is "LLM_FAILURE".
    *   **`store_document_data_task`**: Stores data from `db_storage_payload` into PostgreSQL.
    *   **`pseg_bill_human_review_task` (HUMAN Task):**
        *   Workflow pauses. A human user interacts via the Conductor UI.
        *   **Input to Human:** `human_review_input` (OCR text, original path).
        *   **Expected Human Output (entered in Conductor UI):** A JSON object like `{"corrected_pseg_data_dict": {"account_number": "...", ...}}`.
    *   **`process_human_corrected_data_task`**:
        *   Receives `corrected_pseg_data_dict` from the HUMAN task's output, along with original context like `full_ocr_text` and `document_type_hint`.
        *   Validates the dictionary into `PSEGData`.
        *   Stores the corrected data in PostgreSQL.
*   **`app_worker` Service:** Contains Python logic for all SIMPLE tasks.

### 3.3. Key Libraries and Their Roles
*(No change to this section, as libraries remain the same, their roles in tasks are just orchestrated differently for HITL)*

### 3.4. Handling Custom PDFs (PSEG Bills, Other Complex Bills)
*(No significant change to this section, initial processing is the same, HITL is a fallback)*

## 4. Key Modules (`app/` directory)

*   **`main.py`**: Entry point for Conductor workers. Initializes DB and `TaskHandler`.
*   **`conductor_workers.py`**:
    *   `process_document_task`: Performs LlamaParse PDF to Markdown conversion and then LLM extraction from Markdown.
    *   `store_document_data_task`: Stores data from successful LLM extractions.
    *   `process_human_corrected_data_task`: Processes and store human-corrected data.
*   **`docling_processor.py`**: Handles document parsing (Currently Suspended). Code remains for potential future use.
*   **`llama_parser_processor.py`**: Handles document parsing using LlamaParse.
*   **`llm_extractor.py`**: Works with Markdown input for LLM extraction. (Docling-specific input conversion removed).
*   **`llm_response_models.py`**: No changes.
*   **`config.py`**: `DOCUMENT_PARSER_TYPE` setting removed (or effectively hardcoded to LlamaParse). `LLAMA_CLOUD_API_KEY` is used.
*   **`db_utils.py`**: The `raw_ocr_text` column will now store Markdown content from LlamaParse.

## 5. Technologies Used
*   Python 3.11+
*   pip and `requirements.txt` for dependency management
*   Docker & Docker Compose
*   Netflix Conductor (orchestration)
*   Ollama (local LLM serving)
*   LiteLLM (LLM provider interface)
*   Instructor (structured LLM output)
*   Pydantic (data validation and settings)
*   **LlamaParse & LlamaIndex-core** (document PDF to Markdown conversion)
*   PostgreSQL (database)
*   Loguru (logging - if you choose to integrate it more deeply, currently standard logging)
*   Transformers & SentencePiece (for token counting in LLM input)
*   (Docling library is present in code but suspended from active use)

## 6. Setup and Running

### 6.1. Prerequisites
*   Docker and Docker Compose installed.
*   Python 3.11+ (for local development if not using Docker for everything).
*   Access to an Ollama instance (local or remote) and a LlamaParse API key.

### 6.2. Configuration
1.  **Create `.env` file:**
    Copy `.env.example` (if it exists) to `.env` or create a new `.env` file in the project root.
    Populate it with necessary values:
    ```env
    # Ollama settings
    OLLAMA_BASE_URL=http://ollama:11434 # Or your Ollama instance URL
    OLLAMA_EXTRACTOR_MODEL_NAME=mistral:7b-instruct-q4_K_M # Or your preferred model

    # Database settings
    DATABASE_URL=postgresql://pseguser:psegpassword@postgres_db_service:5432/psegdb

    # Conductor settings
    CONDUCTOR_BASE_URL=http://conductor-server:8080/api

    # LlamaParse API Key (REQUIRED)
    LLAMA_CLOUD_API_KEY="your_llama_parse_api_key_here"
    ```
2.  **Review `app/config.py`:** For default values and other settings.

### 6.3. Building and Running with Docker Compose

1.  **Build the services:**
    ```bash
    docker-compose build
    ```
2.  **Start the services:**
    ```bash
    docker-compose up -d
    ```
    This will start the `app_worker`, `conductor-server`, `postgres_db_service`, and `ollama` (if configured in compose).

### 6.4. Local Development (without full Docker build for app_worker)

If you want to run `app_worker` locally (e.g., for faster debugging of Python code) while other services (Conductor, DB, Ollama) run in Docker:

1.  **Ensure services are up:** `docker-compose up -d conductor-server postgres_db_service ollama` (or whichever backing services you need).
2.  **Install dependencies locally:**
    Create and activate a virtual environment:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```
    Install requirements:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Set Environment Variables:** Ensure your local terminal session has the environment variables defined in your `.env` file (e.g., by using `python-dotenv` in your script or sourcing the `.env` file if your shell supports it, or manually setting them).
4.  **Run the application:**
    ```bash
    python app/main.py
    ```

## 7. How to Run
### Prerequisites
*(No change, but ensure Conductor UI is accessible for HITL steps)*

### Step 1: Build and Start Services
*(No change, but ensure you use `--build` if you modified code like the temporary LLM failure for testing)*

### Step 2: Verify Services
*(No change)*

### Step 3: Register Workflow (if needed) & Test
1.  **Access Conductor UI:** `http://localhost:5002`.
2.  **Register/Update Workflow Definition:** Ensure `document_processing_workflow` (v2) from `ocr_processing_workflow.json` is uploaded/updated.
3.  **Run Workflow (Happy Path & HITL Path):** Follow testing steps outlined in "Phase 2, Step 3 & 4" above.

*(Rest of the sections like Clean Up, Development Notes can remain largely the same, just be mindful that the workflow is now more complex)*