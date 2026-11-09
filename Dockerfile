
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies for PDF OCR
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    poppler-utils \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency list
COPY requirements.txt .

# Install CPU-compatible PyTorch and Python dependencies
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

# Copy project files
COPY . .

# Create runtime directories
RUN mkdir -p uploads outputs/api_results data/vector_db

# Expose FastAPI port
EXPOSE 8000

# Start API
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
