# Advanced Document Intelligence Pipeline with Docling, LLMs, and Conductor

## 1. Introduction and Goals

This project implements a robust, containerized pipeline for advanced intelligent document processing. It leverages:
*   **`docling` library:** For comprehensive document parsing, OCR, and structural analysis.
*   **Large Language Models (LLMs):** For extracting structured information, guided by `instructor`.
*   **Netflix Conductor:** For orchestrating the overall workflow, including a Human-in-the-Loop (HITL) step for LLM extraction failures.
*   **Poetry:** For dependency management.

The workflow processes documents (initially PSEG utility bills) as follows:
1.  **Document Processing Task (`doc_processing_task`):**
    *   Parses the input document using `docling` (`app.docling_processor.py`).
    *   Attempts to extract structured data (e.g., `PSEGData` model) using an LLM (`app.llm_extractor.py`).
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
*   **`docling` Integration:** `docling[vlm]` for parsing and OCR.
*   **LLM Configuration:** Uses Ollama-served models, `instructor`, and `litellm`.
*   **HITL for LLM Extraction:** If `extract_pseg_data` fails (returns `None`), workflow routes to a HUMAN task in Conductor. Human provides corrected JSON via Conductor UI.
*   **Configuration:** Centralized in `app/config.py` (Pydantic `BaseSettings`).
*   **Dependency Management:** Poetry (`pyproject.toml`).

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
        DPTask -- "Input: document_path, <br/> document_type_hint" --> DoclingProc["app.docling_processor.parse_document_with_docling"];
        DoclingProc -- "Output: DoclingDocument" --> LLMExtractor["app.llm_extractor.extract_pseg_data"];
        LLMExtractor -- "Output: Optional[PSEGData]" --> DPTask;
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
        HumanReviewTask -- "Input: human_review_input <br/> (full_ocr_text, original_doc_path, etc.)" --> Human["Human Reviewer <br/> (Uses Conductor UI)"];
        Human -- "Provides output JSON: <br/> { corrected_pseg_data_dict: { ... } }" --> HumanReviewTask;
        HumanReviewTask -- "Output: corrected_pseg_data_dict" --> Workflow;
    end
    
    Workflow -- "Schedules process_human_corrected_data_task" --> AppWorkerService;
    subgraph AppWorkerDBHumanContext ["App Worker: process_human_corrected_data_task Execution"]
        direction LR
        AppWorkerService -.-> ProcessHumanDataTask["process_human_corrected_data_task <br/> (app.conductor_workers.process_human_corrected_data_task)"];
        ProcessHumanDataTask -- "Input: corrected_pseg_data_dict, <br/> full_ocr_text, document_type_hint" --> DBUtilsHuman["app.db_utils.insert_pseg_data"];
        DBUtilsHuman -- "Stores data" --> PGSQLServiceDB;
        ProcessHumanDataTask -- "Output: db_human_corrected_output" --> Workflow;
    end

    %% Common elements referenced earlier in the original diagram
    DoclingProc -- "Uses" --> DoclingLib["Docling Library"];
    LLMExtractor -- "Uses" --> LiteLLMLib["LiteLLM Library"];
    LiteLLMLib -- "Communicates with" --> OllamaService["Ollama Service (LLMs)"];
    LLMExtractor -- "Uses" --> InstructorLib["Instructor Library"];
    LLMExtractor -- "Uses Pydantic Model" --> PSEGDataModel["PSEGData <br/> (app.llm_response_models)"];
    
    subgraph SettingsAndConfigRef ["Shared Configuration"]
      PydanticSettings["app.config.settings"]
    end
    DPTask -- "Reads config" --> PydanticSettings;
    LLMExtractor -- "Reads config" --> PydanticSettings;
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
    class DoclingLib,DoclingCoreLib,LiteLLMLib,InstructorLib extLib;
    class PSEGDataModel,PydanticSettings pydantic;
    class Human human;
```

### 3.2. Component Breakdown

*   **User Input:** Documents submitted via cURL.
*   **Netflix Conductor:** Orchestrates `document_processing_workflow` (v2) from `ocr_processing_workflow.json`.
    *   **`doc_processing_task`**:
        *   Calls `docling` and then LLM extraction.
        *   Outputs `status_doc_processing` ("LLM_SUCCESS" or "LLM_FAILURE").
        *   If "LLM_SUCCESS", outputs `db_storage_payload`.
        *   If "LLM_FAILURE", outputs `human_review_input` (containing `full_ocr_text`, `original_document_path`, `document_type_hint`).
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

*   **`main.py`**: Entry point for Conductor workers. Initializes DB and `TaskHandler` (now registers `process_human_corrected_data_task` as well).
*   **`conductor_workers.py`**:
    *   `process_document_task`: Performs docling and LLM extraction, outputs status for HITL routing.
    *   `store_document_data_task`: Stores data from successful LLM extractions.
    *   `process_human_corrected_data_task`: New task to process and store human-corrected data.
*   **`docling_processor.py`**: No changes.
*   **`llm_extractor.py`**: No changes (failure to extract `None` is the trigger).
*   **`llm_response_models.py`**: No changes.
*   **`config.py`**: No changes.
*   **`db_utils.py`**: No changes (used by both direct and human-corrected storage paths).

## 5. Technologies Used
*(No change to this section)*

## 6. Project Setup and Execution Plan

This section outlines the steps from initial setup to running the full workflow with HITL.

### Phase 0: Migration to Poetry & Initial Setup
*(No change to this section)*

### Phase 1: Core Application Logic Implementation (Docling, LLM, Conductor Workers, DB, HITL Core)

*   **Done:** All original tasks.
*   **Done:** `ocr_processing_workflow.json` (v2) updated for HITL decision logic, including a HUMAN task.
*   **Done:** `app/conductor_workers.py` updated:
    *   `process_document_task` now outputs `status_doc_processing` and conditional payloads.
    *   Added `process_human_corrected_data_task`.
*   **Done:** `app/main.py` registers the new `process_human_corrected_data_task`.

### Phase 2: System Integration, Testing (including HITL), and Refinement

*   **Current:** Docker image build is ongoing/completed.
*   **Next:**
    1.  **Verify Docker Services:** Use `docker-compose ps`. Ensure `app_worker`, `conductor-server`, `conductor-ui`, `ollama`, `postgres_db_service` are running.
    2.  **Register Workflow with Conductor:**
        *   Access Conductor UI (`http://localhost:5002`).
        *   Register `ocr_processing_workflow.json` (v2). **Ensure any previous version is updated or replaced.**
        *   **(Note:** Before running the cURL commands below, create a directory named `sample_documents` in your project root (next to `app/`, `Dockerfile`, etc.) and place your test PDF files (e.g., `your_GOOD_sample_pseg_bill.pdf`, `your_BAD_sample_or_corrupt.pdf`) inside it.)
    3.  **Test Workflow Execution (Happy Path - LLM Success):**
        *   Trigger the `document_processing_workflow` with a document you expect the LLM to process successfully.
        ```bash
            # (Same cURL as before)
            curl -X POST \
              http://localhost:8080/api/workflow/document_processing_workflow \
              -H 'Content-Type: application/json' \
              -d '{
                "document_path": "/usr/src/app/sample_documents/your_GOOD_sample_pseg_bill.pdf",
                "document_type_hint": "PSEG_BILL" 
              }'
            ```
        *   Monitor in Conductor UI: Should go through `doc_processing_task` -> `llm_quality_gate_decision` (LLM_SUCCESS branch) -> `store_document_data_task`.
        *   Verify data in PostgreSQL.
    4.  **Test Workflow Execution (HITL Path - LLM Failure):**
        *   To simulate LLM failure for testing:
            *   **Option A (Code Change):** Temporarily modify `app.llm_extractor.extract_pseg_data` to sometimes (or always) return `None`. Rebuild the `app_worker` image.
            *   **Option B (Difficult Input):** Provide a document that is very different from a PSEG bill, or a corrupted PDF, hoping the LLM fails to extract `PSEGData`.
        *   Trigger the workflow:
        ```bash
            curl -X POST \
              http://localhost:8080/api/workflow/document_processing_workflow \
              -H 'Content-Type: application/json' \
              -d '{
                "document_path": "/usr/src/app/sample_documents/your_BAD_sample_or_corrupt.pdf",
                "document_type_hint": "PSEG_BILL_FOR_FAILURE_TEST"
              }'
            ```
        *   Monitor in Conductor UI:
            *   `doc_processing_task` should complete, outputting `status_doc_processing: "LLM_FAILURE"`.
            *   `llm_quality_gate_decision` should route to the `LLM_FAILURE` branch.
            *   The workflow should pause at `pseg_bill_human_review_task` (state: `IN_PROGRESS`).
        *   **Perform Human Review via Conductor UI:**
            *   Go to the paused workflow instance in Conductor UI.
            *   Find the `pseg_bill_human_review_task`.
            *   View its "Input" tab. You should see `human_review_input` (containing `full_ocr_text`, `original_document_path`, `document_type_hint`).
            *   Use this information to manually construct a valid JSON for `PSEGData`.
            *   Go to the "Actions" tab for the HUMAN task and select "Update Task".
            *   In the "Output Data" field, paste your JSON in the required format:
                ```json
                {
                  "corrected_pseg_data_dict": {
                    "account_number": "MANUAL123",
                    "customer_name": "Manual Entry",
                    "service_address": "123 Human Ln",
                    "billing_address": "PO Box HITL",
                    "billing_date": "2023-01-15",
                    "billing_period_start_date": "2023-01-01",
                    "billing_period_end_date": "2023-01-31",
                    "due_date": "2023-02-01",
                    "total_amount_due": 100.00,
                    "previous_balance": 0.0,
                    "payments_received": 0.0,
                    "current_charges": 100.0,
                    "line_items": [{"description": "Manual charge", "amount": 100.00}],
                    "raw_text_summary": "Manually reviewed and entered."
                  }
                }
                ```
            *   Click "Update". The HUMAN task should complete.
        *   The workflow should then proceed to `process_human_corrected_data_task`.
        *   Verify logs for this task and check that data is stored in PostgreSQL.
    5.  **Refine Prompts and Error Handling:** Based on test results.

### Phase 3: Extending to Other Document Types (Future)
*(No change to this section for now, but HITL makes this more robust)*

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