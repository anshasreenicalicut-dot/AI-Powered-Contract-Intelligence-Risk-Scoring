
import os
import re
import json
from io import BytesIO
from pathlib import Path

import requests
import streamlit as st
import fitz
import pytesseract

from PIL import Image
from docx import Document


# ==========================================================
# CONFIGURATION
# ==========================================================

API_URL = os.getenv(
    "CUAD_API_URL",
    "http://127.0.0.1:8000"
).rstrip("/")

TESSERACT_PATH = os.getenv(
    "TESSERACT_CMD",
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

if Path(TESSERACT_PATH).exists():
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

st.set_page_config(
    page_title="CUAD Contract Intelligence",
    page_icon="📑",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown("""
<style>
.stApp {
    background-color: #f4f7fb;
}

.main-title {
    font-size: 32px;
    font-weight: 700;
    color: #17365D;
}

.subtitle {
    color: #64748b;
    font-size: 15px;
}

div[data-testid="stMetric"] {
    background-color: white;
    padding: 16px;
    border-radius: 12px;
    border: 1px solid #dce4ef;
}

.stButton button {
    border-radius: 8px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)


# ==========================================================
# SESSION STATE
# ==========================================================

DEFAULTS = {
    "uploaded_contracts": [],
    "last_upload": None,
    "analysis": None,
    "contract_text": "",
    "contract_filename": "",
    "copy_message": ""
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ==========================================================
# API FUNCTIONS
# ==========================================================

def check_api():
    try:
        response = requests.get(
            f"{API_URL}/health",
            timeout=5
        )
        response.raise_for_status()
        return True
    except requests.RequestException:
        return False


def upload_contract(uploaded_file):
    files = {
        "file": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            uploaded_file.type
        )
    }

    response = requests.post(
        f"{API_URL}/upload",
        files=files,
        timeout=60
    )

    response.raise_for_status()
    return response.json()


def get_task_status(task_id):
    response = requests.get(
        f"{API_URL}/task/{task_id}",
        timeout=15
    )

    response.raise_for_status()
    return response.json()


def search_contracts(query, top_k):
    response = requests.get(
        f"{API_URL}/search",
        params={
            "query": query,
            "top_k": top_k
        },
        timeout=120
    )

    response.raise_for_status()
    return response.json()


def get_metadata(contract_id):
    response = requests.get(
        f"{API_URL}/metadata/{contract_id}",
        timeout=15
    )

    response.raise_for_status()
    return response.json()


# ==========================================================
# DOCUMENT EXTRACTION
# ==========================================================

def extract_pdf_text(file_bytes):
    pdf = fitz.open(
        stream=file_bytes,
        filetype="pdf"
    )

    text_parts = []

    try:
        for page in pdf:
            page_text = page.get_text("text").strip()

            # OCR fallback for scanned pages
            if not page_text:
                pix = page.get_pixmap(
                    matrix=fitz.Matrix(2, 2),
                    alpha=False
                )

                image = Image.open(
                    BytesIO(pix.tobytes("png"))
                )

                page_text = pytesseract.image_to_string(
                    image
                ).strip()

            text_parts.append(page_text)

    finally:
        pdf.close()

    return "\n".join(text_parts)


def extract_docx_text(file_bytes):
    document = Document(
        BytesIO(file_bytes)
    )

    paragraphs = [
        paragraph.text
        for paragraph in document.paragraphs
        if paragraph.text.strip()
    ]

    return "\n".join(paragraphs)


def extract_contract_text(uploaded_file):
    file_bytes = uploaded_file.getvalue()
    extension = Path(uploaded_file.name).suffix.lower()

    if extension == ".pdf":
        return extract_pdf_text(file_bytes)

    if extension == ".docx":
        return extract_docx_text(file_bytes)

    raise ValueError(
        "Only PDF and DOCX files are supported."
    )


# ==========================================================
# RISK DETECTION RULES
# ==========================================================

RISK_RULES = {
    "Automatic renewal": {
        "weight": 15,
        "patterns": [
            r"automatically renew",
            r"automatic renewal",
            r"auto-renew",
            r"renewal term",
            r"renew automatically"
        ]
    },

    "Confidentiality obligations": {
        "weight": 8,
        "patterns": [
            r"confidential information",
            r"confidentiality obligations",
            r"non-disclosure",
            r"keep.*confidential",
            r"disclose.*confidential"
        ]
    },

    "Payment penalties": {
        "weight": 15,
        "patterns": [
            r"late fee",
            r"late payment penalty",
            r"overdue amounts",
            r"interest on overdue",
            r"penalty for late payment"
        ]
    },

    "Broad indemnification": {
        "weight": 20,
        "patterns": [
            r"indemnify and hold harmless",
            r"defend.*indemnify",
            r"indemnification obligations"
        ]
    },

    "Unlimited liability": {
        "weight": 25,
        "patterns": [
            r"unlimited liability",
            r"without limitation.*liability",
            r"no limitation of liability"
        ]
    },

    "Unilateral termination": {
        "weight": 15,
        "patterns": [
            r"terminate.*at its sole discretion",
            r"terminate.*without cause",
            r"unilateral termination"
        ]
    },

    "Intellectual property ownership": {
        "weight": 10,
        "patterns": [
            r"intellectual property rights",
            r"ownership of intellectual property",
            r"assign.*intellectual property",
            r"work made for hire"
        ]
    },

    "Non-compete obligations": {
        "weight": 12,
        "patterns": [
            r"non-compete",
            r"noncompetition",
            r"restrict.*competing",
            r"not engage in.*competing"
        ]
    },

    "Data protection obligations": {
        "weight": 8,
        "patterns": [
            r"personal data",
            r"data protection",
            r"security breach",
            r"data processing agreement"
        ]
    }
}


# ==========================================================
# IMPORTANT CLAUSE RULES
# ==========================================================

CLAUSE_RULES = {
    "Termination": [
        r"termination",
        r"terminate this agreement",
        r"right to terminate"
    ],

    "Payment Obligations": [
        r"payment terms",
        r"shall pay",
        r"payment obligations",
        r"invoice"
    ],

    "Confidentiality": [
        r"confidential information",
        r"confidentiality",
        r"non-disclosure"
    ],

    "Intellectual Property": [
        r"intellectual property",
        r"ownership of work",
        r"license rights"
    ],

    "Governing Law": [
        r"governing law",
        r"governed by the laws",
        r"jurisdiction"
    ],

    "Indemnification": [
        r"indemnification",
        r"indemnify",
        r"hold harmless"
    ],

    "Limitation of Liability": [
        r"limitation of liability",
        r"liability cap",
        r"maximum liability"
    ],

    "Automatic Renewal": [
        r"automatic renewal",
        r"automatically renew",
        r"renewal term"
    ]
}


# ==========================================================
# EVIDENCE EXTRACTION
# ==========================================================

def find_evidence(text, patterns, max_matches=3):
    sentences = re.split(
        r"(?<=[.!?])\s+|\n+",
        text
    )

    matches = []

    for sentence in sentences:
        sentence = sentence.strip()

        if not sentence:
            continue

        for pattern in patterns:
            if re.search(
                pattern,
                sentence,
                flags=re.IGNORECASE
            ):
                if sentence not in matches:
                    matches.append(sentence)
                break

        if len(matches) >= max_matches:
            break

    return matches


# ==========================================================
# CONTRACT RISK SCORING
# ==========================================================

def analyze_contract(text):
    risk_flags = []
    total_score = 0

    for category, rule in RISK_RULES.items():

        evidence = find_evidence(
            text,
            rule["patterns"]
        )

        if evidence:
            total_score += rule["weight"]

            risk_flags.append({
                "category": category,
                "weight": rule["weight"],
                "evidence": evidence
            })

    score = min(total_score, 100)

    if score <= 30:
        risk_level = "Low"
    elif score <= 60:
        risk_level = "Medium"
    else:
        risk_level = "High"

    important_clauses = {}

    for clause, patterns in CLAUSE_RULES.items():
        matches = find_evidence(
            text,
            patterns,
            max_matches=5
        )

        if matches:
            important_clauses[clause] = matches

    return {
        "risk_score": score,
        "risk_level": risk_level,
        "words_analyzed": len(text.split()),
        "risk_flags": risk_flags,
        "important_clauses": important_clauses,
        "scoring_method": "Rule-based keyword screening"
    }


# ==========================================================
# DISPLAY CONTRACT IDS
# ==========================================================

def display_upload_details():

    last = st.session_state.last_upload

    if not last:
        return

    st.markdown("---")
    st.subheader("Uploaded Contract Details")

    col1, col2 = st.columns(2)

    with col1:
        st.text_input(
            "Contract ID",
            value=last["contract_id"],
            disabled=True,
            key="display_contract_id"
        )

    with col2:
        st.text_input(
            "Task ID",
            value=last["task_id"],
            disabled=True,
            key="display_task_id"
        )

    st.caption(
        f"Filename: {last['filename']}"
    )

    st.download_button(
        "Download Contract IDs",
        data=json.dumps({
            "filename": last["filename"],
            "contract_id": last["contract_id"],
            "task_id": last["task_id"],
            "status": last.get("status", "queued")
        }, indent=4),
        file_name="contract_ids.json",
        mime="application/json"
    )

    st.markdown("### Processing Status")

    if st.button(
        "Check Processing Status",
        key="check_last_task"
    ):

        try:
            result = get_task_status(
                last["task_id"]
            )

            last["status"] = result.get(
                "status",
                "unknown"
            )

            if last["status"] == "completed":
                st.success("Contract processing completed!")

                st.json(
                    result.get("result", result)
                )

            elif last["status"] == "failed":
                st.error(
                    result.get("error", "Processing failed.")
                )

            else:
                st.info(
                    f"Current status: {last['status']}"
                )

        except requests.RequestException as error:
            st.error(str(error))


# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("📑 CUAD Intelligence")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Upload & Risk Analysis",
        "Semantic Search",
        "Contract Metadata",
        "API Status",
        "About Project"
    ]
)

st.sidebar.markdown("---")

if check_api():
    st.sidebar.success("API Connected")
else:
    st.sidebar.error("API Disconnected")

st.sidebar.caption(
    "CUAD Legal Contract Intelligence System"
)


# ==========================================================
# DASHBOARD
# ==========================================================

if page == "Dashboard":

    st.markdown(
        '<div class="main-title">'
        'Contract Intelligence Dashboard'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'AI-powered legal contract analysis and retrieval'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Uploaded Contracts",
            len(st.session_state.uploaded_contracts)
        )

    with col2:
        st.metric(
            "API Status",
            "Online" if check_api() else "Offline"
        )

    with col3:
        st.metric(
            "Words Analyzed",
            (
                st.session_state.analysis["words_analyzed"]
                if st.session_state.analysis
                else 0
            )
        )

    st.markdown("---")
    st.subheader("System Features")

    col1, col2 = st.columns(2)

    with col1:
        st.info("**Contract Upload**\n\nUpload PDF and DOCX files.")
        st.info("**Risk Scoring**\n\nIdentify potential contract risks.")
        st.info("**Semantic Search**\n\nSearch contract passages.")

    with col2:
        st.info("**Clause Identification**\n\nIdentify important clauses.")
        st.info("**Metadata Extraction**\n\nRetrieve contract information.")
        st.info("**Task Monitoring**\n\nTrack processing status.")

    st.markdown("---")
    st.subheader("Uploaded Contract History")

    if st.session_state.uploaded_contracts:
        st.dataframe(
            st.session_state.uploaded_contracts,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No contracts uploaded yet.")

    display_upload_details()


# ==========================================================
# UPLOAD AND RISK ANALYSIS
# ==========================================================

elif page == "Upload & Risk Analysis":

    st.title(
        "📑 AI-Powered Contract Intelligence & Risk Scoring"
    )

    st.write(
        "Upload a contract to extract text, identify risks "
        "and analyze important clauses."
    )

    uploaded_file = st.file_uploader(
        "Upload a PDF or DOCX contract",
        type=["pdf", "docx"],
        key="contract_uploader"
    )

    if uploaded_file:

        st.success(
            f"Selected file: {uploaded_file.name}"
        )

        st.write(
            f"File size: {uploaded_file.size / 1024:.2f} KB"
        )

        if st.button(
            "Analyze Contract",
            type="primary",
            use_container_width=True
        ):

            try:
                with st.spinner(
                    "Extracting and analyzing contract..."
                ):

                    contract_text = extract_contract_text(
                        uploaded_file
                    )

                    if not contract_text.strip():
                        raise ValueError(
                            "No text found in the uploaded document."
                        )

                    analysis = analyze_contract(
                        contract_text
                    )

                st.session_state.contract_text = contract_text
                st.session_state.contract_filename = uploaded_file.name
                st.session_state.analysis = analysis

                st.success("Contract analysis completed!")

            except Exception as error:
                st.error(
                    f"Analysis failed: {error}"
                )

    # Risk results
    if st.session_state.analysis:

        analysis = st.session_state.analysis

        st.markdown("---")
        st.subheader("Contract Risk Assessment")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Risk Score",
                f'{analysis["risk_score"]}/100'
            )

        with col2:
            st.metric(
                "Risk Level",
                analysis["risk_level"]
            )

        with col3:
            st.metric(
                "Words Analyzed",
                analysis["words_analyzed"]
            )

        st.progress(
            analysis["risk_score"] / 100,
            text=f'Risk Score: {analysis["risk_score"]}/100'
        )

        st.markdown("---")
        st.subheader("Potential Risk Flags")

        if analysis["risk_flags"]:

            for flag in analysis["risk_flags"]:

                with st.expander(
                    f'{flag["category"]} '
                    f'({flag["weight"]} points)',
                    expanded=True
                ):

                    for evidence in flag["evidence"]:
                        st.warning(
                            f'Evidence: "{evidence}"'
                        )

        else:
            st.success(
                "No configured risk patterns detected."
            )

        st.markdown("---")
        st.subheader("Important Clauses")

        if analysis["important_clauses"]:

            for clause, matches in (
                analysis["important_clauses"].items()
            ):

                with st.expander(
                    f"{clause} ({len(matches)} matches)"
                ):

                    for match in matches:
                        st.write(f"• {match}")

        else:
            st.info(
                "No configured clause patterns detected."
            )

        st.markdown("---")
        st.subheader("Extracted Contract Text")

        with st.expander("View extracted text"):
            st.text_area(
                "Contract Text",
                value=st.session_state.contract_text,
                height=300
            )

        report = {
            "filename": st.session_state.contract_filename,
            "analysis": analysis,
            "contract_text": st.session_state.contract_text
        }

        st.download_button(
            "Download Contract Analysis JSON",
            data=json.dumps(report, indent=4),
            file_name="contract_analysis.json",
            mime="application/json",
            use_container_width=True
        )

        st.caption(
            "Risk scores are rule-based screening indicators, "
            "not definitive legal assessments."
        )

    # Backend upload
    st.markdown("---")
    st.subheader("Upload Contract to FastAPI")

    if uploaded_file:

        if st.button(
            "Upload to Backend",
            use_container_width=True
        ):

            try:
                with st.spinner("Uploading contract to FastAPI..."):

                    result = upload_contract(
                        uploaded_file
                    )

                last = {
                    "filename": uploaded_file.name,
                    "contract_id": result["contract_id"],
                    "task_id": result["task_id"],
                    "status": result.get("status", "queued")
                }

                st.session_state.last_upload = last

                st.session_state.uploaded_contracts.append({
                    "Filename": uploaded_file.name,
                    "Contract ID": last["contract_id"],
                    "Task ID": last["task_id"],
                    "Status": last["status"]
                })

                st.success(
                    "Contract uploaded successfully!"
                )

            except requests.RequestException as error:
                st.error(
                    f"Backend upload failed: {error}"
                )

            except KeyError as error:
                st.error(
                    f"API response did not contain expected field: {error}"
                )

    display_upload_details()


# ==========================================================
# SEMANTIC SEARCH
# ==========================================================

elif page == "Semantic Search":

    st.title("🔎 Semantic Contract Search")

    query = st.text_area(
        "Enter your search query",
        placeholder="Example: intellectual property rights"
    )

    top_k = st.slider(
        "Number of results",
        min_value=1,
        max_value=20,
        value=5
    )

    if st.button(
        "Search Contracts",
        type="primary",
        use_container_width=True
    ):

        if not query.strip():
            st.warning("Enter a search query.")

        else:

            try:
                with st.spinner("Searching contracts..."):

                    result = search_contracts(
                        query,
                        top_k
                    )

                results = result.get("results", [])

                st.success(
                    f"Search completed: {len(results)} results"
                )

                for index, item in enumerate(results):

                    with st.expander(
                        f"Search Result {index + 1}",
                        expanded=True
                    ):

                        if isinstance(item, dict):
                            st.json(item)
                        else:
                            st.write(item)

                if not results:
                    st.info("No matching passages found.")

            except requests.RequestException as error:
                st.error(
                    f"Search failed: {error}"
                )


# ==========================================================
# CONTRACT METADATA
# ==========================================================

elif page == "Contract Metadata":

    st.title("📋 Contract Metadata")

    contract_id = st.text_input(
        "Enter Contract ID",
        value=(
            st.session_state.last_upload["contract_id"]
            if st.session_state.last_upload
            else ""
        )
    )

    if st.button(
        "Retrieve Metadata",
        use_container_width=True
    ):

        if not contract_id.strip():
            st.warning("Enter a Contract ID.")

        else:

            try:
                metadata = get_metadata(
                    contract_id.strip()
                )

                st.success("Metadata retrieved successfully!")

                st.json(metadata)

                st.download_button(
                    "Download Metadata JSON",
                    data=json.dumps(metadata, indent=4),
                    file_name="contract_metadata.json",
                    mime="application/json"
                )

            except requests.RequestException as error:
                st.error(
                    f"Metadata retrieval failed: {error}"
                )


# ==========================================================
# API STATUS
# ==========================================================

elif page == "API Status":

    st.title("⚙️ API Monitoring")

    if check_api():

        st.success("FastAPI server is running.")

        try:
            response = requests.get(
                f"{API_URL}/",
                timeout=5
            )

            st.json(response.json())

        except requests.RequestException as error:
            st.error(str(error))

    else:

        st.error(
            "FastAPI server is not reachable."
        )

        st.code(
            "python -m uvicorn src.api.main:app --reload"
        )

    st.write("API Base URL")
    st.code(API_URL)

    st.markdown(
        f"[Open Swagger API Documentation]({API_URL}/docs)"
    )


# ==========================================================
# ABOUT PROJECT
# ==========================================================

elif page == "About Project":

    st.title("About CUAD Contract Intelligence")

    st.markdown("""
    ## Project Overview

    CUAD Contract Intelligence is an NLP-assisted legal
    contract analysis and information retrieval system.

    ## Technologies Used

    - Python
    - DistilBERT
    - PyTorch
    - spaCy
    - Sentence Transformers
    - ChromaDB
    - FastAPI
    - Streamlit
    - PyMuPDF
    - Tesseract OCR

    ## Main Modules

    1. Contract upload
    2. PDF and DOCX extraction
    3. Scanned PDF OCR
    4. Contract risk screening
    5. Important clause identification
    6. Semantic search
    7. Metadata retrieval
    8. Processing task tracking

    ## Dataset

    Contract Understanding Atticus Dataset (CUAD)

    Number of contracts: 510

    ## Disclaimer

    Automated risk flags and scores are intended for
    preliminary screening and must be reviewed by
    qualified legal professionals.
    """)

    st.markdown("---")

    st.caption(
        "CUAD Legal Contract Intelligence System | Week 4"
    )
