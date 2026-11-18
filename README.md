# AI-Powered-Contract-Intelligence-Risk-Scoring

An AI-powered legal document analysis and retrieval system built using the **Contract Understanding Atticus Dataset (CUAD)**. The project combines Natural Language Processing (NLP), Transformer-based Question Answering, Named Entity Recognition (NER), semantic search, vector databases and REST API development to help users analyze legal contracts.

---

## 1. Project Overview

Legal contracts contain important information about parties, termination conditions, payment obligations, intellectual property rights, confidentiality and other legal clauses. Manually reviewing large numbers of contracts can be time-consuming.

The CUAD Legal Contract Intelligence System aims to simplify this process through machine learning and NLP.

The system supports:

- Legal contract dataset preparation and preprocessing.
- PDF and DOCX document ingestion.
- PDF text extraction and OCR for scanned documents.
- Transformer-based extractive Question Answering.
- Named Entity Recognition for organizations, dates and monetary values.
- Sentence Transformer-based text embeddings.
- ChromaDB vector storage.
- Semantic search across contract passages.
- FastAPI-based backend services.
- Streamlit-based user interface.
- Docker-based deployment.

## 2. Project Objectives

1. Process and prepare the CUAD dataset for machine learning.
2. Develop a baseline NLP pipeline for legal entity extraction.
3. Train a DistilBERT-based Question Answering model.
4. Extract text from digital and scanned PDF documents.
5. Convert contract text into numerical embeddings.
6. Store and retrieve contract passages using ChromaDB.
7. Develop REST APIs for contract upload, search and metadata retrieval.
8. Build an interactive Streamlit frontend.
9. Containerize the application using Docker.

## 3. Dataset

**Dataset:** Contract Understanding Atticus Dataset (CUAD)

The CUAD dataset contains legal contracts with expert annotations covering important contractual provisions.

Dataset information:

| Attribute | Details |
|---|---|
| Dataset | CUAD |
| Number of contracts | 510 |
| Processed examples | 47,542 |
| Data format | JSON / JSONL |
| Main application | Legal contract understanding |
| Model task | Extractive Question Answering |

The processed data is organized into training, validation and testing datasets.

### Dataset processing

The preprocessing pipeline includes:

1. Loading the original CUAD JSON files.
2. Extracting contract titles and paragraphs.
3. Processing questions and annotated answers.
4. Preparing training-ready JSONL records.
5. Splitting the data into training, validation and testing sets.
6. Saving processed data for model training and evaluation.

The QA dataset is split at the contract level to avoid placing the same contract in both training and validation datasets.

## 4. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| PyTorch | Deep learning framework |
| Hugging Face Transformers | DistilBERT Question Answering |
| spaCy | Named Entity Recognition |
| Sentence Transformers | Text embeddings |
| ChromaDB | Vector database |
| FastAPI | Backend API |
| Streamlit | Frontend application |
| PyMuPDF | PDF text extraction |
| PyPDF | PDF processing |
| Tesseract OCR | Scanned document text recognition |
| pdf2image | PDF page-to-image conversion |
| Pandas | Data processing |
| Scikit-learn | Data preparation and evaluation |
| Docker | Application containerization |

## 5. Project Architecture

The application is organized into four main development stages.

### Week 1 – Data Parsing and Baseline Modeling

- CUAD dataset collection and inspection.
- JSON and JSONL data preparation.
- PDF text extraction.
- Tesseract-based OCR pipeline.
- Baseline spaCy NER development.

### Week 2 – Transformer-Based Question Answering

- DistilBERT model preparation.
- Tokenization and answer-span alignment.
- Handling answerable and unanswerable questions.
- Model training and validation.
- Evaluation using Exact Match and token-level F1.

### Week 3 – Embeddings and Semantic Search

- Sentence Transformer integration.
- 384-dimensional text embeddings using `all-MiniLM-L6-v2`.
- Contract chunking.
- ChromaDB vector storage.
- Similarity-based semantic retrieval.
- FastAPI backend development.

### Week 4 – Application and Deployment

- Streamlit frontend development.
- FastAPI and frontend integration.
- Dockerfile and Docker Compose configuration.
- Application testing and documentation.

## 6. Project Structure

```text
CUAD/
│
├── cuad_download/
│   ├── CUADv1.json
│   ├── train_separate_questions.json
│   └── test.json
│
├── data/
│   ├── pdfs/
│   ├── processed/
│   │   ├── contracts.jsonl
│   │   ├── cuad_examples.jsonl
│   │   └── cuad_summary.json
│   │
│   ├── qa/
│   │   ├── train.jsonl
│   │   ├── validation.jsonl
│   │   ├── test.jsonl
│   │   └── summary.json
│   │
│   ├── vector_db/
│   └── week2/
│
├── models/
│   ├── cuad_qa_v1/
│   ├── cuad_clause_classifier/
│   └── cuad_ner/
│
├── outputs/
│   ├── api_results/
│   ├── evaluation/
│   ├── evaluation_test/
│   ├── evaluation_validation/
│   ├── extracted_text/
│   ├── ner_results/
│   └── qa_evaluation/
│
├── src/
│   ├── api/
│   │   └── main.py
│   │
│   ├── vector/
│   │   ├── embeddings.py
│   │   └── vector_store.py
│   │
│   ├── ner_pipeline.py
│   ├── pdf_processor.py
│   ├── pdf_processor_ocr.py
│   ├── prepare_data.py
│   ├── prepare_qa_data.py
│   ├── train_qa.py
│   ├── train_transformer.py
│   ├── evaluate_qa.py
│   └── predict_clause.py
│
├── uploads/
│
├── app.py
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── requirements.txt
└── README.md
```

*Note: This is a logical project structure based on the developed modules. Some folders and model artifacts are generated during execution. The NER model directory is required only after successful training.*

## 7. Installation and Environment Setup

### Prerequisites

- Windows 10 or Windows 11.
- Python 3.9.
- Visual Studio Code.
- Git.
- Docker Desktop (for container deployment).
- Tesseract OCR.
- Poppler (for PDF-to-image conversion).

### Step 1: Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd CUAD
```

Alternatively, open the existing project directory in VS Code.

### Step 2: Create a virtual environment

```powershell
py -3.9 -m venv .venv
```

### Step 3: Activate the environment

```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 4: Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If Streamlit or Requests is not already included:

```powershell
pip install streamlit requests
```

## 8. Week 1 – Data Preparation and OCR

### Dataset preparation

The CUAD dataset is processed into JSONL format for downstream model training.

Run the appropriate preparation script:

```powershell
python -m src.prepare_data
```

For the Question Answering dataset:

```powershell
python -m src.prepare_qa_data
```

The generated files are stored in:

```text
data/processed/
data/qa/
```

### PDF text extraction

Digital PDFs can be processed using PyMuPDF and PyPDF.

### OCR pipeline

The OCR pipeline uses:

- Tesseract OCR for text recognition.
- pdf2image for converting PDF pages into images.
- PyMuPDF for extracting existing digital text.

Install the required Python packages:

```powershell
pip install pytesseract pdf2image Pillow PyMuPDF
```

Install Tesseract OCR and Poppler separately on Windows.

Example Tesseract path:

```text
C:\Program Files\Tesseract-OCR\tesseract.exe
```

Example Poppler path:

```text
C:\poppler\Library\bin
```

Run the OCR pipeline:

```powershell
python -m src.pdf_processor_ocr `
    --input-dir ".\data\pdfs" `
    --output-dir ".\outputs\extracted_text" `
    --tesseract-cmd "C:\Program Files\Tesseract-OCR\tesseract.exe" `
    --poppler-path "C:\poppler\Library\bin" `
    --dpi 250
```

The actual executable paths must match the local installation.

Expected output:

```text
outputs/
└── extracted_text/
    ├── contract_name.txt
    └── contract_name.json
```

The OCR pipeline should be tested with both digital PDFs and scanned PDFs.

## 9. Week 2 – DistilBERT Question Answering

The project uses the DistilBERT model for extractive Question Answering.

**Pretrained model:** `distilbert/distilbert-base-uncased`

### Model configuration

| Parameter | Value |
|---|---|
| Architecture | DistilBERT |
| Maximum sequence length | 384 |
| Training epochs | 2 |
| Batch size | 2 |
| Training device | CPU |
| Answer extraction | Start and end token prediction |

### Training

The QA training script is:

```text
src/train_qa.py
```

Run:

```powershell
python -m src.train_qa
```

The trained model is stored in:

```text
models/cuad_qa_v1/
```

### Training results

The recorded training run produced:

| Metric | Result |
|---|---:|
| Training loss | 1.2011 |
| Validation loss | 1.2562 |
| Training runtime | Approximately 2 hours 27 minutes |
| Evaluation runtime | Approximately 13 minutes |

These are training and validation losses, not final test accuracy.

### Model evaluation

Run:

```powershell
python -m src.evaluate_qa
```

Evaluation includes:

- Exact Match (EM).
- Token-level F1.
- Category-wise performance.
- Answer extraction performance.

Evaluation output:

```text
outputs/
└── qa_evaluation/
    ├── qa_predictions.jsonl
    └── qa_metrics.json
```

Final test EM and F1 should be reported from the generated metrics file after evaluation finishes successfully.

## 10. Week 3 – Embeddings and Vector Database

### Sentence Transformers

The project uses:

`sentence-transformers/all-MiniLM-L6-v2`

This model converts text passages into 384-dimensional vector embeddings.

### Embedding generation

The embedding module is:

```text
src/vector/embeddings.py
```

The vector store module is:

```text
src/vector/vector_store.py
```

### ChromaDB integration

ChromaDB stores contract passage embeddings and supports similarity-based retrieval.

The general workflow is:

1. Extract contract text.
2. Split text into manageable chunks.
3. Generate embeddings.
4. Store vectors and metadata in ChromaDB.
5. Convert user queries into embeddings.
6. Retrieve the most relevant contract passages.

The vector database location is:

```text
data/vector_db/
```

Actual persistence depends on the configuration in `vector_store.py`.

## 11. Week 3 – FastAPI Backend

FastAPI provides REST API endpoints for contract upload, processing, semantic search and metadata retrieval.

### Start the backend

```powershell
python -m uvicorn src.api.main:app --reload
```

Backend URL:

```text
http://127.0.0.1:8000
```

Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

### API endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API information |
| GET | `/health` | Health check |
| POST | `/upload` | Upload PDF or DOCX contract |
| GET | `/task/{task_id}` | Retrieve processing status |
| GET | `/search` | Semantic contract search |
| GET | `/metadata/{contract_id}` | Retrieve contract metadata |

### Contract upload

The upload endpoint accepts PDF and DOCX files.

Example response:

```json
{
    "message": "Contract uploaded successfully",
    "task_id": "generated-task-id",
    "contract_id": "generated-contract-id",
    "status": "queued"
}
```

The task ID is generated dynamically for every upload.

### Processing workflow

```text
Contract Upload
      |
      v
File Validation
      |
      v
Text Extraction
      |
      v
Text Chunking
      |
      v
Embedding Generation
      |
      v
ChromaDB Storage
      |
      v
Metadata Generation
      |
      v
Processing Completed
```

**Note:** The current API implementation uses PyPDF for PDF text extraction. Scanned PDF OCR requires integration of the OCR pipeline into the upload processing workflow.

The API's `/search` endpoint performs semantic retrieval through the vector store. It should not be confused with DistilBERT extractive Question Answering.

## 12. Week 4 – Streamlit Frontend

The Streamlit application provides a graphical interface for interacting with the FastAPI backend.

Run the application:

```powershell
python -m streamlit run app.py
```

Default application URL:

```text
http://localhost:8501
```

### Frontend modules

#### Dashboard

Displays:
- Number of uploaded contracts during the current session.
- API connection status.
- Main application features.

#### Upload Contract

Allows users to:
- Upload PDF files.
- Upload DOCX files.
- Submit contracts for processing.
- View generated task IDs and contract IDs.
- Check processing status.

#### Semantic Search

Allows users to:
- Enter natural-language search queries.
- Select the number of results.
- Retrieve relevant contract passages.

#### Contract Metadata

Allows users to:
- Enter a contract ID.
- Retrieve contract metadata.
- Download metadata in JSON format.

#### API Status

Displays:
- API connection status.
- Backend information.
- Swagger documentation link.

#### About Project

Provides information about the CUAD dataset, technologies and system architecture.

The Streamlit application communicates with FastAPI using HTTP requests.

## 13. Week 4 – Docker Deployment

The project includes:

```text
Dockerfile
docker-compose.yml
.dockerignore
```

These files are intended to support containerization.

### Prerequisites

Install Docker Desktop and ensure that its Linux container engine is running.

Verify:

```powershell
docker --version
docker compose version
docker info
```

### Build containers

```powershell
docker compose build
```

### Start services

```powershell
docker compose up -d
```

### Check running containers

```powershell
docker compose ps
```

### View logs

```powershell
docker compose logs -f
```

### Stop services

```powershell
docker compose down
```

**Important:** Docker deployment is complete only after the image builds successfully, the containers start and the application endpoints are tested. Docker Desktop must be running before these commands are executed.

## 14. Testing and Validation

The following tests should be performed before final submission.

| Test | Expected result |
|---|---|
| Dataset preprocessing | JSONL files generated |
| QA model loading | Trained DistilBERT model loads |
| QA evaluation | EM and F1 metrics generated |
| Digital PDF extraction | Text extracted successfully |
| Scanned PDF OCR | OCR text generated |
| NER prediction | Organizations, dates and money detected |
| Embedding generation | 384-dimensional vectors generated |
| ChromaDB indexing | Contract chunks stored |
| Semantic search | Relevant passages retrieved |
| API health | Healthy response |
| Contract upload | Task and contract IDs returned |
| Task tracking | Processing status returned |
| Metadata retrieval | Contract metadata returned |
| Streamlit frontend | All pages load and API calls work |
| Docker deployment | Containers build and start successfully |

## 15. Limitations

The following considerations apply to the current implementation:

1. OCR quality depends on PDF resolution, scan quality and document formatting.
2. NER performance depends on the quality and quantity of annotated training examples.
3. DistilBERT QA performance must be assessed using final evaluation metrics.
4. Semantic search retrieves relevant passages but does not independently generate legal conclusions.
5. The current API task store is in-memory and does not persist task status across server restarts.
6. The current API does not automatically invoke OCR for scanned PDFs.
7. The current semantic search endpoint is separate from the trained DistilBERT QA model.
8. Docker deployment and complete end-to-end integration require successful runtime testing.

This system is intended for research and educational purposes and should not replace professional legal review.

## 16. Future Enhancements

Potential improvements include:

- Integrating OCR directly into the API upload pipeline.
- Connecting DistilBERT QA to the Streamlit interface.
- Adding question-based contract analysis.
- Implementing persistent task tracking.
- Improving NER performance with more annotated legal documents.
- Adding authentication and user management.
- Supporting additional document formats.
- Improving search ranking and retrieval evaluation.
- Deploying the application to a cloud environment.
- Adding downloadable contract analysis reports.

## 17. Project Development Summary

| Week | Development focus |
|---|---|
| Week 1 | Data parsing, preprocessing, PDF extraction, OCR and baseline NER |
| Week 2 | DistilBERT Question Answering model training and evaluation |
| Week 3 | Text embeddings, ChromaDB and FastAPI |
| Week 4 | Streamlit frontend, Dockerization, testing and documentation |

## 18. Author

**Project:** CUAD – AI-Powered Legal Contract Intelligence System

**Core technologies:** Python, PyTorch, DistilBERT, spaCy, Sentence Transformers, ChromaDB, FastAPI, Streamlit and Docker.

---

*Developed as an AI/NLP project focused on intelligent legal contract understanding, information extraction and semantic retrieval.*

## Disclaimer

This project is intended for educational and research purposes only. Predictions may be incorrect or incomplete and must not be relied upon as legal advice or as the sole basis for legal decisions.
