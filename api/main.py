
import uuid
import shutil
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

from pypdf import PdfReader
from docx import Document

from src.vector.vector_store import add_document, search_documents

BASE_DIR = Path(__file__).resolve().parents[2]
UPLOAD_DIR = BASE_DIR / "uploads"
RESULT_DIR = BASE_DIR / "outputs" / "api_results"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="CUAD Contract Intelligence API",
    description="Contract upload, semantic search and metadata extraction",
    version="1.0.0"
)

TASKS = {}


def extract_text(file_path):
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        reader = PdfReader(str(file_path))
        return "\n".join(
            page.extract_text() or "" for page in reader.pages
        )

    if suffix == ".docx":
        document = Document(str(file_path))
        return "\n".join(
            paragraph.text for paragraph in document.paragraphs
        )

    raise ValueError("Unsupported file type")


def split_text(text, chunk_size=1200, overlap=200):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def process_contract(task_id, contract_id, file_path):
    try:
        TASKS[task_id]["status"] = "processing"

        text = extract_text(file_path)

        if not text.strip():
            raise ValueError(
                "No text found. The document may require OCR."
            )

        chunks = split_text(text)

        for index, chunk in enumerate(chunks):
            add_document(contract_id, chunk, index)

        result = {
            "contract_id": contract_id,
            "filename": file_path.name,
            "text_length": len(text),
            "number_of_chunks": len(chunks),
            "metadata": {
                "filename": file_path.name,
                "character_count": len(text)
            }
        }

        result_path = RESULT_DIR / f"{contract_id}.json"
        import json

        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4)

        TASKS[task_id]["status"] = "completed"
        TASKS[task_id]["result"] = result

    except Exception as error:
        TASKS[task_id]["status"] = "failed"
        TASKS[task_id]["error"] = str(error)


@app.get("/")
def home():
    return {
        "message": "CUAD Contract Intelligence API",
        "version": "1.0.0"
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/upload")
async def upload_contract(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    suffix = Path(file.filename).suffix.lower()

    if suffix not in [".pdf", ".docx"]:
        raise HTTPException(
            status_code=400,
            detail="Only PDF and DOCX files are supported."
        )

    contract_id = str(uuid.uuid4())
    task_id = str(uuid.uuid4())

    file_path = UPLOAD_DIR / f"{contract_id}{suffix}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    TASKS[task_id] = {
        "status": "queued",
        "contract_id": contract_id
    }

    background_tasks.add_task(
        process_contract,
        task_id,
        contract_id,
        file_path
    )

    return {
        "message": "Contract uploaded successfully",
        "task_id": task_id,
        "contract_id": contract_id,
        "status": "queued"
    }


@app.get("/task/{task_id}")
def task_status(task_id: str):
    if task_id not in TASKS:
        raise HTTPException(status_code=404, detail="Task not found")

    return TASKS[task_id]


@app.get("/search")
def search(query: str, top_k: int = 5):
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query is required")

    if top_k < 1 or top_k > 20:
        raise HTTPException(
            status_code=400,
            detail="top_k must be between 1 and 20"
        )

    results = search_documents(query, top_k)

    return {
        "query": query,
        "results": results
    }


@app.get("/metadata/{contract_id}")
def get_metadata(contract_id: str):
    import json

    result_path = RESULT_DIR / f"{contract_id}.json"

    if not result_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Contract metadata not found"
        )

    with open(result_path, "r", encoding="utf-8") as f:
        return json.load(f)
