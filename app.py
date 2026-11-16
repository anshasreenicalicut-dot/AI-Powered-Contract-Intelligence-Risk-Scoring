
import os
import time
import requests
import streamlit as st

# ==========================================
# CONFIGURATION
# ==========================================

API_URL = os.getenv(
    "CUAD_API_URL",
    "http://127.0.0.1:8000"
)

st.set_page_config(
    page_title="CUAD Contract Intelligence",
    page_icon="📑",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# CUSTOM CSS
# ==========================================

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

    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #dce4ef;
    }

    .stButton button {
        background-color: #17365D;
        color: white;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# API FUNCTIONS
# ==========================================

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


# ==========================================
# SESSION STATE
# ==========================================

if "uploaded_contracts" not in st.session_state:
    st.session_state.uploaded_contracts = []

if "last_upload" not in st.session_state:
    st.session_state.last_upload = None


# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("📑 CUAD Intelligence")

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Upload Contract",
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

st.sidebar.caption("CUAD Legal Contract Intelligence System")


# ==========================================
# DASHBOARD
# ==========================================

if page == "Dashboard":

    st.markdown(
        '<div class="main-title">Contract Intelligence Dashboard</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">AI-powered legal contract analysis and retrieval</div>',
        unsafe_allow_html=True
    )

    st.write("")

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
            "Available Features",
            "4"
        )

    st.markdown("---")

    st.subheader("System Features")

    feature1, feature2 = st.columns(2)

    with feature1:
        st.info(
            "**Contract Upload**\n\n"
            "Upload PDF and DOCX legal contracts."
        )

        st.info(
            "**Semantic Search**\n\n"
            "Search contract passages using natural language."
        )

    with feature2:
        st.info(
            "**Metadata Extraction**\n\n"
            "View uploaded contract information."
        )

        st.info(
            "**Processing Status**\n\n"
            "Monitor contract processing tasks."
        )

    st.markdown("---")

    st.subheader("Recent Uploads")

    if st.session_state.uploaded_contracts:

        st.dataframe(
            st.session_state.uploaded_contracts,
            use_container_width=True
        )

    else:
        st.info("No contracts uploaded during this session.")


# ==========================================
# UPLOAD CONTRACT
# ==========================================

elif page == "Upload Contract":

    st.title("📤 Upload Legal Contract")

    st.write(
        "Upload a PDF or DOCX contract for text extraction, "
        "chunking and semantic indexing."
    )

    uploaded_file = st.file_uploader(
        "Choose a contract",
        type=["pdf", "docx"]
    )

    if uploaded_file:

        st.success(
            f"Selected file: {uploaded_file.name}"
        )

        st.write(
            f"File size: {uploaded_file.size / 1024:.2f} KB"
        )

        if st.button(
            "Upload and Process Contract",
            use_container_width=True
        ):

            try:

                with st.spinner("Uploading contract..."):

                    result = upload_contract(uploaded_file)

                task_id = result["task_id"]
                contract_id = result["contract_id"]

                st.session_state.last_upload = {
                    "task_id": task_id,
                    "contract_id": contract_id,
                    "filename": uploaded_file.name
                }

                st.session_state.uploaded_contracts.append({
                    "Filename": uploaded_file.name,
                    "Contract ID": contract_id,
                    "Task ID": task_id,
                    "Status": "Queued"
                })

                st.success("Contract uploaded successfully!")

                st.write("**Task ID**")

                st.code(task_id)

                st.write("**Contract ID**")

                st.code(contract_id)

            except requests.RequestException as error:

                st.error(f"Upload failed: {error}")

    st.markdown("---")

    st.subheader("Processing Status")

    task_id_input = st.text_input(
        "Enter Task ID",
        placeholder="Paste your task ID here"
    )

    if st.button("Check Processing Status"):

        if not task_id_input.strip():

            st.warning("Please enter a task ID.")

        else:

            try:

                result = get_task_status(
                    task_id_input.strip()
                )

                status = result.get("status", "unknown")

                if status == "completed":
                    st.success("Contract processing completed!")

                elif status == "failed":
                    st.error("Contract processing failed.")

                else:
                    st.info(f"Current status: {status}")

                st.json(result)

            except requests.RequestException as error:

                st.error(f"Unable to retrieve task: {error}")

    if st.button("Wait for Last Upload to Finish"):

        last_upload = st.session_state.last_upload

        if not last_upload:

            st.warning("Upload a contract first.")

        else:

            task_id = last_upload["task_id"]
            progress = st.progress(0)
            status_text = st.empty()

            try:

                for attempt in range(150):

                    result = get_task_status(task_id)

                    status = result.get("status", "unknown")

                    status_text.write(
                        f"Processing status: {status}"
                    )

                    progress.progress(
                        min((attempt + 1) / 150, 1.0)
                    )

                    if status == "completed":

                        st.success(
                            "Contract processing completed!"
                        )

                        st.json(result.get("result", result))
                        break

                    if status == "failed":

                        st.error(
                            result.get("error", "Processing failed.")
                        )
                        break

                    time.sleep(2)

                else:

                    st.warning(
                        "Processing is taking longer than expected. "
                        "Check the task status again."
                    )

            except requests.RequestException as error:

                st.error(f"Status check failed: {error}")


# ==========================================
# SEMANTIC SEARCH
# ==========================================

elif page == "Semantic Search":

    st.title("🔎 Semantic Contract Search")

    st.write(
        "Search uploaded contract passages using natural language."
    )

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
        use_container_width=True
    ):

        if not query.strip():

            st.warning("Please enter a search query.")

        else:

            try:

                with st.spinner("Searching contract database..."):

                    result = search_contracts(
                        query,
                        top_k
                    )

                results = result.get("results", [])

                st.success(
                    f"Search completed. Results returned: {len(results)}"
                )

                if results:

                    for index, item in enumerate(results):

                        with st.expander(
                            f"Result {index + 1}",
                            expanded=True
                        ):

                            if isinstance(item, dict):

                                st.json(item)

                            else:

                                st.write(item)

                else:

                    st.info("No matching contract passages found.")

            except requests.RequestException as error:

                st.error(f"Search failed: {error}")


# ==========================================
# CONTRACT METADATA
# ==========================================

elif page == "Contract Metadata":

    st.title("📋 Contract Metadata")

    st.write(
        "Retrieve information about a previously uploaded contract."
    )

    contract_id = st.text_input(
        "Enter Contract ID",
        placeholder="Paste contract ID here"
    )

    if st.button(
        "Retrieve Metadata",
        use_container_width=True
    ):

        if not contract_id.strip():

            st.warning("Please enter a contract ID.")

        else:

            try:

                with st.spinner("Retrieving metadata..."):

                    metadata = get_metadata(
                        contract_id.strip()
                    )

                st.success("Metadata retrieved successfully!")

                st.subheader("Contract Information")

                st.json(metadata)

                st.download_button(
                    "Download Metadata JSON",
                    data=__import__("json").dumps(
                        metadata,
                        indent=4
                    ),
                    file_name=f"{contract_id}.json",
                    mime="application/json"
                )

            except requests.RequestException as error:

                st.error(f"Metadata retrieval failed: {error}")


# ==========================================
# API STATUS
# ==========================================

elif page == "API Status":

    st.title("⚙️ API Monitoring")

    if check_api():

        st.success("CUAD FastAPI server is running.")

        try:

            response = requests.get(
                f"{API_URL}/",
                timeout=5
            )

            st.json(response.json())

        except requests.RequestException as error:

            st.error(str(error))

    else:

        st.error("FastAPI server is not reachable.")

        st.code(
            "python -m uvicorn src.api.main:app --reload"
        )

    st.markdown("---")

    st.write("API Base URL")

    st.code(API_URL)

    st.write("Swagger Documentation")

    st.markdown(
        f"[Open API Documentation]({API_URL}/docs)"
    )


# ==========================================
# ABOUT PROJECT
# ==========================================

elif page == "About Project":

    st.title("About CUAD Contract Intelligence")

    st.markdown("""
    ### Project Overview

    CUAD Contract Intelligence is an AI-powered legal contract
    analysis and retrieval system.

    ### Technologies Used

    - Python
    - FastAPI
    - Streamlit
    - DistilBERT
    - Sentence Transformers
    - ChromaDB
    - PyMuPDF / PyPDF
    - OCR tools (Tesseract)

    ### Main Modules

    1. Contract upload
    2. PDF and DOCX text extraction
    3. Document chunking
    4. Text embeddings
    5. Vector database storage
    6. Semantic search
    7. Contract metadata retrieval

    ### Model Information

    Trained model: DistilBERT

    Dataset: CUAD (Contract Understanding Atticus Dataset)

    Number of contracts: 510
    """)
    
    st.markdown("---")

    st.caption(
        "CUAD Contract Intelligence System | Week 4 Development"
    )
