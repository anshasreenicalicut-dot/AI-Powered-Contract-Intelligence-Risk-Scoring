# AI-Powered-Contract-Intelligence-Risk-Scoring

## Project Overview

The **CUAD Legal Contract Clause Classification** project uses Natural Language Processing (NLP) and transformer-based deep learning to analyze legal contract text and predict the category of a clause.

The project uses the **Contract Understanding Atticus Dataset (CUAD)**, a dataset developed for identifying and understanding important clauses in legal contracts. It includes data preparation, text preprocessing, model training, evaluation, and prediction.

The trained model can accept a contract clause as input and return a predicted category with a confidence score.

## Objectives

- Process and prepare legal contract data.
- Preprocess contract text for NLP.
- Fine-tune a transformer-based text classification model.
- Predict potential contract clause categories.
- Evaluate model performance using standard classification metrics.
- Build a reusable prediction pipeline for legal text analysis.

## Technologies Used

- **Programming Language:** Python
- **IDE:** Visual Studio Code
- **NLP / Deep Learning:** Hugging Face Transformers, PyTorch
- **Dataset Processing:** Hugging Face Datasets, Pandas
- **Evaluation:** Scikit-learn
- **Dataset:** Contract Understanding Atticus Dataset (CUAD)
- **Version Control:** Git and GitHub

## Project Structure

```text
CUAD/
├── data/
│   ├── pdfs/
│   ├── processed/
│   │   ├── cuad_examples.jsonl
│   │   ├── contracts.jsonl
│   │   └── cuad_summary.json
│   └── week2/
│       ├── train.jsonl
│       ├── validation.jsonl
│       └── test.jsonl
├── models/
│   └── cuad_clause_classifier/
├── src/
│   ├── prepare_data.py
│   ├── pdf_processor.py
│   ├── ner_pipeline.py
│   ├── prepare_week2_data.py
│   ├── train_transformer.py
│   ├── evaluate_model.py
│   └── predict_clause.py
├── outputs/
├── requirements.txt
├── .gitignore
└── README.md
```

*Note: Some folders and generated files are created only after running the relevant scripts. File names may vary depending on your project configuration.*

## Dataset

This project uses the Contract Understanding Atticus Dataset (CUAD), which contains legal contracts annotated for important contract provisions.

**Dataset resources:**

- CUAD GitHub repository: https://github.com/TheAtticusProject/cuad
- CUAD dataset page: https://huggingface.co/datasets/theatticusproject/cuad

Follow the dataset's licensing and usage terms when downloading or redistributing its contents.

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/CUAD.git
cd CUAD
```

Replace `YOUR-USERNAME` with your GitHub username.

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, follow your organization's execution policy or use the VS Code Command Prompt terminal to activate the environment.

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

PyTorch must be installed with a version compatible with your Python environment. If required, install a compatible CPU version using the official PyTorch installation instructions.

## Workflow

### Step 1: Prepare the dataset

Run the data preparation script:

```powershell
python .\src\prepare_data.py
```

This step prepares the CUAD annotations and generates processed data files.

### Step 2: Prepare Week 2 training data

```powershell
python .\src\prepare_week2_data.py
```

The resulting data is split into training, validation, and test sets. When possible, splitting by contract prevents clauses from the same contract from appearing in multiple sets.

### Step 3: Train the transformer model

```powershell
python .\src\train_transformer.py
```

For a small smoke test, if your script supports these arguments:

```powershell
python .\src\train_transformer.py --data-dir .\data\week2 --output-dir .\models\cuad_clause_classifier --epochs 1 --batch-size 1 --max-train-samples 100 --cpu
```

The smoke test checks whether the training pipeline runs; it is not a final model evaluation.

### Step 4: Evaluate the model

```powershell
python .\src\evaluate_model.py
```

Check your script's `--help` output if your evaluation script requires a model directory or dataset path.

### Step 5: Predict a contract clause

Example:

```powershell
python .\src\predict_clause.py --model-dir .\models\cuad_clause_classifier --text "Either party may terminate this agreement by providing thirty days written notice."
```

Example output:

```json
{
  "text": "Either party may terminate this agreement by providing thirty days written notice.",
  "predicted_category": "Example predicted category",
  "confidence": 0.75,
  "threshold": 0.5,
  "accepted": true
}
```

The values above illustrate the output format only. Actual predictions and confidence scores depend on your trained model.

To require a higher confidence threshold:

```powershell
python .\src\predict_clause.py --model-dir .\models\cuad_clause_classifier --text "This agreement shall be governed by the laws of California." --threshold 0.7
```

## Model Evaluation

The model should be evaluated using:

- **Accuracy:** Overall proportion of correct predictions.
- **Precision:** Proportion of predicted categories that are correct.
- **Recall:** Proportion of relevant examples correctly identified.
- **F1 Score:** Harmonic mean of precision and recall.
- **Confidence threshold:** Minimum confidence used to accept a prediction.

Report the actual evaluation results produced by your script. Do not interpret a successful prediction or training run as proof of high accuracy.

## Important Dataset Consideration

CUAD is primarily designed for extractive question answering: identifying relevant text spans in contracts in response to questions about legal provisions.

Converting CUAD into a conventional clause-classification dataset requires a carefully validated mapping between contract text, questions, answer spans, and category labels. Incorrect mappings can produce misleading predictions and evaluation scores.

This project is an educational NLP prototype, not a substitute for review by a qualified legal professional.

## Future Enhancements

- Develop a Streamlit or Flask interface for uploading contract files.
- Extract text directly from PDF documents.
- Highlight relevant contract passages.
- Add batch processing for multiple contracts.
- Export prediction results to CSV or Excel.
- Improve dataset preparation and validate labels.
- Add model explainability and more comprehensive evaluation.

## Author

**Project:** CUAD Legal Contract Clause Classification  
**Domain:** Natural Language Processing, Deep Learning, Legal AI  
**Language:** Python

## Disclaimer

This project is intended for educational and research purposes only. Predictions may be incorrect or incomplete and must not be relied upon as legal advice or as the sole basis for legal decisions.
