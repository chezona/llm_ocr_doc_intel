# LLM-Aided PSEG Utility Bill Processing Pipeline

## Introduction

This project implements a robust, containerized pipeline specifically designed for processing **PSEG utility bills** (submitted as images or PDFs). It leverages Optical Character Recognition (OCR) for text extraction. The core processing is managed by **LangGraph**, orchestrating a **2-pass sequence of steps**: 
1.  OCR and optional LLM-based text enhancement.
2.  An initial LLM-based classification to determine the document type.
3.  Conditional routing to a specialized LLM-based extraction process if classified as a PSEG bill, or a generic handler for other types. 

LangGraph's stateful nature is key to managing this multi-step flow and enabling iterative refinement. The extracted information, conforming to a PSEG-specific data model, is then stored in a PostgreSQL database. The overall workflow execution is managed by Netflix Conductor, which triggers a single worker that, in turn, runs the LangGraph pipeline. The system is managed with Docker and Docker Compose.

Key components and flow:
1.  **Input**: PSEG bill (image or PDF) via local path or URL.
2.  **Conductor Workflow**: A Netflix Conductor workflow (`ocr_processing_workflow`) is initiated.
3.  **LangGraph Pipeline Worker**: A single Conductor worker (`LangGraphPipelineWorker`) executes the main processing logic by invoking a LangGraph.
    *   **OCR Node**: Performs OCR using **Tesseract OCR** (via `app.ocr_utils`).
    *   **Text Enhancement Node (Optional)**: If configured, an LLM (via `app.llm_text_enhancer`) attempts to correct/improve the raw OCR text.
    *   **Initial Classification Node**: An LLM (via `app.llm_utils.classify_document_type`) classifies the document (e.g., 'pseg_bill', 'invoice', 'receipt', 'other').
    *   **Conditional Routing**: Based on the classification, the graph routes to the appropriate extraction node.
    *   **PSEG Extraction Node**: If classified as 'pseg_bill', an Ollama-hosted LLM (e.g., `mistral:7b-instruct-q4_K_M`) is used with the `instructor` library (via `app.llm_utils.process_ocr_text_with_llm` using a PSEG-specific prompt) to extract structured data into a `PSEGData` Pydantic model.
    *   **Prepare Other Output Node**: Handles documents not classified as PSEG bills.
    *   **Database Preparation Node**: Prepares the extracted/processed data for storage.
4.  **Storage**: The output from the LangGraph (specifically the prepared database payload) is passed to a `DatabaseStorageWorker` (another Conductor task), which stores the structured `PSEGData` and raw OCR text in a PostgreSQL database if identified as a PSEG bill.

## Current Status & Key Findings

*   **Primary Focus**: The pipeline is tuned for processing **PSEG utility bills**.
*   **Orchestration**: The core OCR-Enhance-Classify-Extract logic is managed by a **2-Pass LangGraph pipeline** (defined in `app.graph_orchestrator.py`), invoked by a single Conductor worker (`LangGraphPipelineWorker` in `app.worker.py`).
*   **Improved Extraction for PSEG Bills**: The 2-pass architecture, with an initial classification step followed by a specialized PSEG extraction step using a detailed prompt, is implemented to address previous challenges with null field extraction.
*   **OCR**: **Tesseract OCR remains the default** for text extraction.
*   **LLM Configuration**: Using `mistral:7b-instruct-q4_K_M` (or user-defined alternatives via `.env`) for classification, text enhancement, and extraction, with `instructor` and `litellm` for reliable JSON output. Specific models can be set for classifier (`OLLAMA_CLASSIFIER_MODEL_NAME`), enhancer (`OLLAMA_TEXT_ENHANCER_MODEL_NAME`), and extractor (`OLLAMA_EXTRACTOR_MODEL_NAME`).
*   **Overall Pipeline**: The Conductor workflow (`ocr_processing_workflow.json`) defines `langgraph_pipeline_task` -> `db_storage_task`.

## Implemented 2-Pass LangGraph Architecture for Enhanced Field Extraction

Previously, while document *classification* as `pseg_bill` was successful, the extraction of specific field *values* into the `PSEGData` model often resulted in `null` entries. This indicated that a single-pass LLM call struggled with the combined complexity of classification and detailed field extraction from OCR text.

To address this robustly, a **2-pass data extraction process within the LangGraph pipeline has been implemented**:

1.  **`ocr_node`**: Performs OCR using Tesseract.
2.  **`text_enhancement_node`**: (Optional) Enhances OCR text using an LLM.
3.  **`initial_classification_node`**:
    *   **Action**: Uses an LLM (configured by `OLLAMA_CLASSIFIER_MODEL_NAME`) with a prompt focused *primarily on document classification* ('pseg_bill', 'invoice', 'receipt', 'other').
    *   **Pydantic Model**: `SimpleClassificationOutput`.
    *   **Output**: `classification_result` (containing `document_type` and `confidence`).
4.  **`conditional_routing_node` (Edge Logic)**:
    *   **Action**: Based on `classification_result.document_type`, routes to a specialized extraction node.
5.  **`pseg_extraction_node`** (if classified as `pseg_bill`):
    *   **Action**: Uses an LLM (configured by `OLLAMA_EXTRACTOR_MODEL_NAME`) with a prompt *highly specialized and optimized for PSEG bills*. This prompt includes strong instructions for field population and the detailed PSEG few-shot example.
    *   **Pydantic Model**: `LLMResponseModel` (expecting `PSEGData` in `structured_data`).
    *   **Output**: `final_extracted_data_model` (containing an `ExtractedData` object with `PSEGData`).
6.  **`prepare_other_output_node`** (if not classified as `pseg_bill` or if PSEG extraction fails to identify as PSEG):
    *   **Action**: Creates a standard `ExtractedData` object indicating the document type is 'other' or the initially classified type, with a note.
    *   **Output**: `final_extracted_data_model`.
7.  **`prepare_db_payload_node`**:
    *   **Action**: Consolidates information from `final_extracted_data_model` into the `db_storage_payload` structure expected by the `DatabaseStorageWorker`.

This 2-pass approach offers:
*   **Focused LLM Calls**: Each LLM call has a narrower, more manageable task (classification vs. specific extraction).
*   **Optimized Prompts**: Prompts are tailored for classification (`classify_document_type` in `llm_utils.py`) versus detailed PSEG extraction (`DocumentProcessor` prompt in `llm_utils.py`).
*   **Improved Accuracy**: By breaking down the problem, higher accuracy is expected in both classification and final field extraction for PSEG bills.
*   **Flexibility**: LangGraph allows for easy addition of more steps or specialized nodes for other document types (e.g., `invoice_extraction_node`) in the future.

This architecture is now in place to achieve more reliable and detailed PSEG bill data extraction.

## Architecture Overview

The system is composed of several services that work together:

*   **App Worker (`app_worker` service):** A Python application defining Conductor workers. The key worker is now `LangGraphPipelineWorker`, which runs the document processing graph. The `DatabaseStorageWorker` handles DB insertion.
*   **Conductor Server (`conductor-server` service):** Netflix Conductor server.
*   **Conductor UI (`conductor-ui` service):** Web interface for Conductor.
*   **Ollama (`ollama` service):** Provides access to local LLMs used by the LangGraph nodes.
*   **PostgreSQL (`postgres_db` service):** Database for storing data.
*   **Redis (`redis` service):** Used by Conductor.

### Architectural Diagram (Reflects Implemented LangGraph 2-Pass Architecture)

```mermaid
flowchart TD
    UserInput[\"User Input <br/> (Image URL/Path via cURL)\"] --> CW_API[\"Conductor API <br/> (POST /api/workflow/ocr_processing_workflow)\"];

    subgraph \"Conductor Server & UI (Services)\"
        CW_API --> CS[\"Conductor Server <br/> (conductor_server_service <br/> Port: 8080)\"];
        CS <--> CUI[\"Conductor UI <br/> (conductor_ui_service <br/> Port: 5002)\"];
        CS --> Workflow[\"ocr_processing_workflow <br/> (ocr_processing_workflow.json)\"];
    end

    Workflow -- \"Schedules langgraph_pipeline_task\" --> AppWorkerService[\"App Worker <br/> (app_worker service)\"];

    subgraph \"App Worker: LangGraphPipelineWorker Execution\"
        direction LR
        AppWorkerService -.-> LGWP[\"LangGraphPipelineWorker <br/> (app.worker.LangGraphPipelineWorker)\"];
        LGWP -- \"image_path_or_url, source_type, configs\" --> GraphOrchestrator[\"app.graph_orchestrator.run_graph_pipeline()\"];
        
        subgraph \"LangGraph Pipeline (app.graph_orchestrator - 2-Pass Architecture)\"
            direction TB
            GraphOrchestrator --> OCRNode[\"OCR Node <br/> (Tesseract via ocr_utils)\"];
            OCRNode -- \"Raw OCR Text\" --> EnhanceNode[\"Text Enhancement Node (Optional) <br/> (LLM via llm_text_enhancer)\"];
            EnhanceNode -- \"Corrected/Raw OCR Text\" --> InitialClassifierNode[\"Initial Classification Node <br/> (LLM via llm_utils.classify_document_type)\"];
            InitialClassifierNode -- \"Classification Result\" --> Router{\"Conditional Routing <br/> (route_based_on_classification)\"};
            Router -- \"pseg_bill\" --> PSEGExtractorNode[\"PSEG Extraction Node <br/> (LLM via llm_utils.process_ocr_text_with_llm)\"];
            Router -- \"other/invoice/receipt\" --> PrepareOtherOutputNode[\"Prepare Other Output Node\"];
            PSEGExtractorNode -- \"Extracted PSEGData Model\" --> DBPrepNode[\"DB Preparation Node\"];
            PrepareOtherOutputNode -- \"Generic ExtractedData Model\" --> DBPrepNode;
            DBPrepNode -- \"DB Storage Payload\" --> GraphOrchestrator;
        end
        
        GraphOrchestrator -- \"Output: db_storage_payload\" --> LGWP;
    end
    
    LGWP -- \"Output for db_storage_task\" --> Workflow; 
    %% Back to Conductor workflow context
    Workflow -- \"Schedules db_storage_task\" --> AppWorkerService;
    %% AppWorkerService picks up the next task

    subgraph \"App Worker: DatabaseStorageWorker Execution\"
        direction LR
        AppWorkerService -.-> DBStorageTaskWorker[\"DatabaseStorageWorker <br/> (app.worker.DatabaseStorageWorker)\"];
        DBStorageTaskWorker -- \"document_type, raw_text, structured_info, summary\" --> DBUtilsModule[\"app.db_utils module\"];
        DBUtilsModule -- \"insert_pseg_data()\" --> PGSQLServiceDB[\"PostgreSQL DB <br/> (postgres_db_service)\"];
        PGSQLServiceDB -- \"Stores Data\" --> DBUtilsModule;
        DBUtilsModule -- \"db_id / status\" --> DBStorageTaskWorker;
    end

    classDef default fill:#fff,stroke:#333,stroke-width:2px,color:#333;
    classDef userInput fill:#f9f,stroke:#333,stroke-width:2px;
    classDef conductor fill:#lightblue,stroke:#333,stroke-width:2px;
    classDef appWorker fill:#lightgreen,stroke:#333,stroke-width:2px;
    classDef services fill:#orange,stroke:#333,stroke-width:2px;
    classDef database fill:#cyan,stroke:#333,stroke-width:2px;
    classDef pipelineGraphStyle fill:#cornsilk,stroke:#333,stroke-width:1.5px;

    class UserInput userInput;
    class CW_API,CS,CUI,Workflow conductor;
    class AppWorkerService,LGWP,DBStorageTaskWorker appWorker;
    class GraphOrchestrator,OCRNode,EnhanceNode,InitialClassifierNode,Router,PSEGExtractorNode,PrepareOtherOutputNode,DBPrepNode pipelineGraphStyle;
    class DBUtilsModule module;
    class PGSQLServiceDB services;
```

## Core Workflow: `ocr_processing_workflow` (v2 with LangGraph)

The primary document processing flow, `ocr_processing_workflow` (defined in `ocr_processing_workflow.json`), now consists of:

1.  **`langgraph_pipeline_task`**:
    *   **Input**: `image_path_or_url`, `source_type` (from workflow input).
    *   **Action**: Managed by `app.worker.LangGraphPipelineWorker`. This worker invokes the LangGraph pipeline defined in `app.graph_orchestrator.run_graph_pipeline()`. The graph internally performs a 2-pass process:
        *   **Pass 1: OCR & Classification**
            *   OCR: Using Tesseract via `app.ocr_utils`.
            *   Text Enhancement (Optional): Using an LLM via `app.llm_text_enhancer`.
            *   Initial Classification: Using an LLM via `app.llm_utils.classify_document_type` to determine `document_type`.
        *   **Pass 2: Specialized Extraction (Conditional)**
            *   If 'pseg_bill': Detailed extraction using an LLM with `instructor` via `app.llm_utils.process_ocr_text_with_llm` (PSEG-specific prompt) to get structured `PSEGData`.
            *   If 'other' (or other types): Handled by `prepare_other_output_node`.
        *   **DB Payload Preparation**: Formatting data for the next step by `prepare_db_payload_node`.
    *   **Output**: A dictionary payload including `document_type`, `raw_text`, `structured_info`, and `summary`.

2.  **`db_storage_task`**:
    *   **Input**: `document_type`, `raw_text`, `structured_info`, `summary` (from `langgraph_pipeline_task` output).
    *   **Action**: Managed by `app.worker.DatabaseStorageWorker`. Stores data into PostgreSQL using `app.db_utils` if `document_type` is `pseg_bill`.
    *   **Output**: `db_storage_status`, `db_id`.
    *   **Current Status**: Works as before, now consuming output from the LangGraph pipeline worker.

## Key Modules and Their Roles (`app/` directory)

*   **`graph_orchestrator.py`**: Defines the LangGraph pipeline, including its state, nodes (OCR, text enhancement, initial classification, PSEG extraction, other type preparation, DB preparation), and the 2-pass graph execution logic. The stateful nature of LangGraph is crucial for managing the conditional flow.
*   **`worker.py`**: Defines Conductor workers.
    *   `LangGraphPipelineWorker`: Invokes the 2-pass LangGraph pipeline from `graph_orchestrator.py`. It now configures and passes `classifier_llm_config` in addition to other configs to the graph.
    *   `DatabaseStorageWorker`: Stores results in the database.
*   **`ocr_utils.py`**: Contains `AdvancedOCRProcessor`. Handles PDF to image, image preprocessing. Primary OCR is Tesseract. Called by the OCR node in the LangGraph.
*   **`llm_text_enhancer.py`**: (Refactored) Provides `enhance_ocr_text` function for optional OCR correction using an LLM. Called by the text enhancement node in the LangGraph.
*   **`llm_utils.py`**: Contains:
    *   `classify_document_type`: A new function using a focused prompt and `SimpleClassificationOutput` model for the first pass classification.
    *   `DocumentProcessor`: Class used by `process_ocr_text_with_llm`. Its prompt has been specifically enhanced with **few-shot examples for PSEG bills** to improve extraction into `PSEGData`.
    *   `process_ocr_text_with_llm`: Now accepts `model_name` as an argument and is used for the second pass PSEG-specific extraction.
    *   Uses `instructor` with `litellm` for structured data extraction.
*   **`llm_response_models.py`**: Defines `LLMResponseModel` (used by `DocumentProcessor` for PSEG extraction, expecting `PSEGData`) and the new `SimpleClassificationOutput` (used by `classify_document_type`).
*   **`models.py`**: Defines `PSEGData` and `ExtractedData`.
*   **`config.py`**: Central configuration. Added `OLLAMA_CLASSIFIER_MODEL_NAME` (defaults to `OLLAMA_EXTRACTOR_MODEL_NAME`), in addition to `OLLAMA_EXTRACTOR_MODEL_NAME` and `OLLAMA_TEXT_ENHANCER_MODEL_NAME`.
*   **`db_utils.py`**: Manages database schema and data insertion.
*   **`config_models.py`**: Defines Pydantic models for `OCRConfig` and `LLMConfig`.

## Technologies Used

*   **Python 3.10+**
*   **Docker & Docker Compose**
*   **Netflix Conductor**: For workflow orchestration.
*   **Ollama**: For serving LLMs (e.g., `moondream:latest` for text analysis).
*   **Tesseract OCR**: Primary engine for text extraction from documents.
*   **`instructor` library (v1.0+)**: For reliable structured JSON output from LLMs, ensuring adherence to Pydantic models.
*   **`litellm` library**: As an interface layer for `instructor` to communicate with the Ollama-served LLM.
*   **Pydantic**: For data validation, settings management, and defining LLM response schemas.
*   **PostgreSQL**: Relational database.
*   **`pdf2image`**: For converting PDF pages to images.
*   **`OpenCV` (python-headless)**: For image preprocessing.
*   **`httpx`**: For asynchronous HTTP calls (e.g., in `ocr_utils.py` for Ollama health checks).
*   **`requests`**: For synchronous HTTP calls (e.g., in `conductor_utils.py`, and `ocr_utils.py` for fetching URL content).

## Setup and Running the Application

### Prerequisites

1.  **Docker and Docker Compose:** Ensure they are installed.
2.  **Git:** For cloning repositories.
3.  **Cloned Conductor Repository:**
    *   The Conductor server and UI are built from source. Clone the official Conductor repository *alongside* this project's directory. If this project is at `your_workspace/llm_ocr_doc_intel`, clone Conductor to `your_workspace/conductor`:
        ```bash
        # In your_workspace/ (i.e., the parent directory of llm_ocr_doc_intel)
        git clone https://github.com/conductor-oss/conductor.git
        ```
4.  **Ollama Models:**
    *   Ensure the LLMs specified in `app/config.py` are pulled into your Ollama instance. These include:
        *   `OLLAMA_EXTRACTOR_MODEL_NAME` (default: `mistral:7b-instruct-q4_K_M`) - for PSEG data extraction.
        *   `OLLAMA_TEXT_ENHANCER_MODEL_NAME` (default: `mistral:7b-instruct-q4_K_M`) - for optional text enhancement.
        *   `OLLAMA_CLASSIFIER_MODEL_NAME` (defaults to the extractor model, e.g., `mistral:7b-instruct-q4_K_M`) - for initial document classification.
        *   `OLLAMA_VISION_MODEL_NAME` (default: `moondream:latest`) - if Ollama-based OCR is ever enabled (currently Tesseract is primary).
    *   Example pull commands: 
        `ollama pull mistral:7b-instruct-q4_K_M`
        `ollama pull moondream:latest`
5.  **Tesseract OCR:** The `app_worker` Docker image installs Tesseract. No separate host installation needed for Dockerized workflow.

### Running the System

1.  **Navigate to the project directory:**
    ```bash
    cd path/to/your/llm_ocr_doc_intel
    ```

2.  **Build and start all services:**
    This builds images, including Conductor from your local clone. The first build might take some time.
    ```bash
    docker-compose up -d --build
    ```
    *   To view logs for a specific service (e.g., the app worker): `docker-compose logs -f app_worker`
    *   To view all logs: `docker-compose logs -f`

3.  **Register the Workflow Definition with Conductor:**
    (Only needed once, or if `ocr_processing_workflow.json` changes).
    Wait for `conductor-server` to be healthy (check `docker-compose ps` or logs: `docker-compose logs conductor-server`). It might take a minute or two for Conductor to fully initialize.
        ```bash
        curl -X POST -H "Content-Type: application/json" -H "Accept: application/json" \
             --data @ocr_processing_workflow.json \
             http://localhost:8080/api/metadata/workflow | cat
        ```
    A 409 status code (`{"status":409,"message":"Workflow with ocr_processing_workflow.1 already exists!"...}`) means it's already registered, which is fine. An empty response or success message also indicates success.

### Starting a Workflow (Testing)

Use `curl` to submit a document for processing. The `PSEG2.pdf` file is included in the repository for testing.
Workflow inputs are now `image_path_or_url` and `source_type`.

*   **Example for the local `PSEG2.pdf` file:**
    The `docker-compose.yml` mounts the project root to `/usr/src/app/mounted_project_files` inside the `app_worker` container.
    ```bash
    curl -X POST -H "Content-Type: application/json" \
         -d '{ "image_path_or_url": "/usr/src/app/mounted_project_files/PSEG2.pdf", "source_type": "path" }' \
         http://localhost:8080/api/workflow/ocr_processing_workflow | cat
    ```
    The output of this command is the Workflow Instance ID.

*   **For a document accessible via URL:**
    ```bash
    curl -X POST -H "Content-Type: application/json" \
         -d '{ "image_path_or_url": "YOUR_IMAGE_OR_PDF_URL_HERE", "source_type": "url" }' \
         http://localhost:8080/api/workflow/ocr_processing_workflow | cat
    ```

### Accessing Conductor UI

*   **URL**: `http://localhost:5002`
    Use the Workflow Instance ID from the `curl` command to find and inspect your workflow run (e.g., search under "Workflow Executions"). Check the input/output of each task (`ocr_task`, `llm_processing_task`, `db_storage_task`) and the overall workflow status.

### Checking Database Contents

To manually check what was stored in the database for a given workflow:
1.  The `db_storage_task` logs will indicate if data for a specific type (like `pseg_bill` or `receipt`) was attempted to be stored and the resulting `db_id`. If the `llm_processing_task` classifies the document as "other" (e.g., due to an LLM error/timeout), specific tables like `pseg_bills` will not be populated for that run.

2.  To connect directly using `psql` from the `postgres_db_service` container:
    ```bash
    docker exec -it postgres_db_service psql -U pseguser -d psegdb
    ```
    Once connected, you can run SQL queries:
    ```sql
    -- To see tables
    \dt

    -- To see columns of pseg_bills
    \d pseg_bills;
    -- To see columns of receipts
    \d receipts;
    \d receipt_items;
    \d merchants;

    -- To select data (example, use actual ID if available from logs/Conductor)
    SELECT * FROM pseg_bills WHERE id = YOUR_DB_ID;
    SELECT * FROM receipts WHERE id = YOUR_DB_ID; 
    ```

## Troubleshooting & Current Challenges

*   **LLM Timeout with `moondream:latest` for Text Analysis:**
    *   **Symptom:** `llm_processing_task` returns `document_type: 'other'` and an error message in `structured_info` indicating `litellm.Timeout` after 120 seconds (current default for `OLLAMA_REQUEST_TIMEOUT`) during an `instructor` retry.
    *   **Cause:** `moondream:latest` appears to be too slow or struggles to generate the complex JSON required by `LLMResponseModel` from detailed OCR text within the configured timeout. The model is relatively small and primarily vision-focused.
    *   **Potential Mitigations:**
        1.  **Increase `OLLAMA_REQUEST_TIMEOUT`:** In `app/config.py`, try a significantly larger value (e.g., `300` or `600` seconds) and rebuild the `app_worker` image (`docker-compose up -d --build app_worker`).
        2.  **Use a More Capable LLM for Text Analysis:** This is the **recommended approach if timeouts persist**. Change `OLLAMA_MULTIMODAL_MODEL_NAME` in `app/config.py` to a more powerful text processing model available in your Ollama instance (e.g., `llama3`, `mistral:7b-instruct`, or other models known for strong instruction following and JSON generation). Ensure it's pulled into Ollama (e.g., `ollama pull llama3`). Rebuild `app_worker`.
        3.  **Simplify Prompt/Schema:** If `moondream:latest` must be used, try reducing the number of fields or complexity in `LLMResponseModel` and the prompt in `app/llm_utils.py`.
        4.  **Adjust `instructor` Retries:** `max_retries` in `app/llm_utils.py` is currently 2. If the base model is too slow, multiple retries exacerbate the timeout. Consider setting to 0 or 1 during timeout troubleshooting.

### Advanced Workflow Inspection via API

While the Conductor UI (`http://localhost:5002`) is the primary tool for visualizing workflows, there might be situations (like UI issues or for automation) where direct API calls are useful. The Conductor server runs on `http://localhost:8080`.

Remember to replace `YOUR_WORKFLOW_INSTANCE_ID` with the actual ID you receive when starting a workflow.

*   **Get Workflow Instance Details (replace `YOUR_WORKFLOW_INSTANCE_ID`):**
    ```bash
    curl -X GET http://localhost:8080/api/workflow/YOUR_WORKFLOW_INSTANCE_ID?includeTasks=true | jq . | cat
    ```
    Look for the `langgraph_pipeline_ref` task to see its input/output, and similarly for `db_task_ref`.

*   **Get Task Logs (Conductor internal logs, not application logs):**
    This endpoint provides Conductor's internal logging about task execution attempts, not the Python application's `logger` output.
    ```bash
    curl -X GET http://localhost:8080/api/queue/logs/YOUR_WORKFLOW_INSTANCE_ID/YOUR_TASK_ID | cat
    ```
    You'd find `YOUR_TASK_ID` from the workflow instance details above (e.g., the `taskId` associated with `langgraph_pipeline_ref`).
    For detailed application logs (Python logger output), refer to `docker-compose logs -f app_worker`.

## Future Enhancements

*   **Experiment with Different LLMs:** Test more robust LLMs (e.g., Llama 3, Mixtral, larger variants) for the text analysis task to improve reliability, speed, and accuracy of structured data extraction.
*   **Refine `LLMResponseModel` and Prompts:** Further tailor prompts and Pydantic models per document type.
*   **Expanded Document Type Support.**
*   **Improved Data Validation & Confidence Scoring:** Implement post-extraction validation beyond Pydantic schema and add confidence scores.
*   **Batch Processing, User Interface, Async DB Ops, Enhanced Logging/Monitoring, Security.**
*   **Explore LangGraph:** For more complex, agentic sub-workflows within tasks (e.g., iterative OCR correction if a high-quality vision model is used, or multi-stage analysis for very complex documents).

## Development Log & Key Changes

This section summarizes the key steps, challenges, and resolutions during the development and debugging of the LLM-Aided Document Processing Pipeline.

*   **Initial Setup**: Goal: OCR, LLM extraction, Conductor orchestration.
*   **Model Evolution & Ollama Integration**: Explored `moondream`, `gemma3:1b`, settled on `mistral:7b-instruct-q4_K_M` for stability.
*   **Timeout Management**: Addressed LLM timeouts by increasing `OLLAMA_REQUEST_TIMEOUT`.
*   **Code Debugging (`app/llm_utils.py`)**: Resolved `IndentationError`, `InstructorRetryException` (adopted `mode=instructor.Mode.JSON`), `AttributeError` (ensured `ExtractedData` always returned).
*   **Workflow Orchestration (Conductor)**: Achieved end-to-end workflow runs; confirmed DB persistence; cleaned stuck workflows.
*   **Conductor UI Issues**: Diagnosed UI proxy errors; confirmed API functionality.
*   **Database Interaction**: Corrected `psql` usage; verified data storage.
*   **Refactoring to LangGraph (Initial)**: Shifted core OCR-Enhance-Extract logic from multiple Conductor tasks to a single LangGraph pipeline invoked by `LangGraphPipelineWorker` for better flexibility.
*   **`llm_text_enhancer.py` Integration**: Refactored and integrated the existing text enhancer into the LangGraph pipeline as an optional node.
*   **LangGraph Debugging & Configuration**: Resolved `ImportError`s (model name inconsistencies, standardized to `OLLAMA_EXTRACTOR_MODEL_NAME`, `OLLAMA_TEXT_ENHANCER_MODEL_NAME`, `OLLAMA_VISION_MODEL_NAME`), `ModuleNotFoundError` (`langgraph`), `TypeError` in graph node, and Ollama 404 for enhancer model (aligned model versions).
*   **PSEG Bill Extraction Improvement (Attempt 1 - Prompt Engineering)**: Updated prompt in `app/llm_utils.py` DocumentProcessor with few-shot PSEG example. Achieved correct classification (`document_type='pseg_bill'`) but `structured_data` fields remained null.
*   **Implemented 2-Pass LangGraph Architecture (Current)**:
    *   **Objective**: Address null field extraction for PSEG bills robustly.
    *   **Changes**:
        *   Modified `app/graph_orchestrator.py`: Updated `DocumentGraphState`, added new nodes (`initial_classification_node`, `pseg_extraction_node`, `prepare_other_output_node`), redefined graph edges for 2-pass logic (OCR -> Enhance -> Classify -> Route -> PSEG_Extract/Other_Prep -> DB_Prep).
        *   Modified `app/llm_utils.py`: Added `classify_document_type` function with a focused classification prompt and `SimpleClassificationOutput` model. Modified `process_ocr_text_with_llm` to take `model_name` for PSEG-specific extraction pass. Introduced `_get_ollama_model_identifier` helper.
        *   Modified `app/llm_response_models.py`: Added `SimpleClassificationOutput`.
        *   Modified `app/config.py`: Added `OLLAMA_CLASSIFIER_MODEL_NAME` (defaults to extractor model).
        *   Modified `app/worker.py`: `LangGraphPipelineWorker` now creates and passes `classifier_llm_config` (using `OLLAMA_CLASSIFIER_MODEL_NAME`) to the graph pipeline.
    *   **Expected Outcome**: Improved accuracy in PSEG field extraction by separating classification and specialized extraction tasks.