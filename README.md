# File-Based Multimodal RAG for Test Case Generation

This project is a robust, multi-modal Retrieval Augmented Generation (RAG) application designed to automate the creation of QA test cases. By ingesting Product Requirement Documents (PRDs), API Specifications, and UI screenshots, it leverages advanced Large Language Models (LLMs) to generate comprehensive test suites, covering positive, negative, and boundary scenarios.


- **Demo Video:** [Click](https://drive.google.com/file/d/1jWWxH_QJYnLCdLA5nYyXUyBvDT4LN6Zp/view) 



## Architecture

The application follows a modular architecture designed for scalability and maintainability.

1.  **Ingestion Layer (`src/services/ingestion.py`)**:
    *   **Text Processing**: Standard loaders for PDF, DOCX, TXT, MD, JSON, YAML.
    *   **Visual Processing**: **Groq Vision Models** (`llama-4-maverick`, `llama-4-scout`) analyze UI screenshots and convert them into detailed technical descriptions.
    *   **Chunking**: Recursive character splitting for optimal context retrieval.

2.  **Retrieval Layer (`src/services/retrieval.py`)**:
    *   **Vector Database**: **ChromaDB** stores semantic embeddings locally.
    *   **Embeddings**: **Hugging Face** (`all-MiniLM-L6-v2`) runs locally for fast, private, and cost-effective embedding generation.

3.  **Generation Layer (`src/services/generation.py`)**:
    *   **LLM Engine**: **Groq** (`llama-3.3-70b-versatile`, `gpt-oss-120b`) generates structured test cases.
    *   **Guardrails**: **Llama Guard** ensures input/output safety and relevance.
    *   **Structured Output**: Pydantic models enforce strict JSON schemas for consistent reporting.

4.  **User Interface (`src/ui/streamlit_app.py`)**:
    *   A clean, minimal **Streamlit** dashboard for configuration, file upload, and interaction.

## Tools & Technology Stack

*   **Language**: Python 3.10+
*   **LLM Provider**: [Groq](https://groq.com/) (Ultra-low latency inference)
*   **Vector Store**: ChromaDB
*   **Frameworks**: LangChain, Streamlit
*   **Embeddings**: Sentence-Transformers (Hugging Face)
*   **IDEs Used**: VS Code, Cursor (for AI-assisted development)

## Setup & Installation

### Prerequisites
*   Python 3.10 or higher installed.
*   A **Groq API Key** (Get it from [console.groq.com](https://console.groq.com/keys)).

### Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/K-Tikaraya/File-Based-Multimodal-RAG-for-Test-Case-Use-Case-Generation.git
    cd rag_project
    ```

2.  **Create a Virtual Environment**:
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # Mac/Linux
    source venv/bin/activate
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
   

4.  **Environment Configuration**:
    Create a `.env` file in the root directory:
    ```env
    GROQ_API_KEY=gsk_ your_api_key_here
    ```


5  **Add rag_data_source folder**:
    In root directory build a folder named rag_data_source to store uploaded folder locally

## Running the Application

Launch the interface using Streamlit:

```bash
streamlit run src/ui/streamlit_app.py
```

The application will open in your default browser at `http://localhost:8501`.

## Usage Example

1.  **Configure Models** (Sidebar):
    *   Select a Vision Model (e.g., *Maverick*).
    *   Select a Generation Model (e.g., *Llama 3.3 70B*).
    *   Enable Guardrails (Optional).

2.  **Upload Knowledge Base**:
    *   Upload your **PRD** (`.md`, `.docx`), **API Spec** (`.yaml`), and **UI Mockups** (`.png`).
    *   Click **Ingest Files**. The artifacts are processed and stored in the vector database.

3.  **Generate Test Cases**:
    *   Enter a query: *"Generate test cases for the User Signup flow including edge cases for password validation."*
    *   Click **Generate**.

4.  **View Results**:
    *   The app will display a structured Test Suite with:
        *   **Goal**: What is being tested.
        *   **Preconditions**: Required state.
        *   **Test Data**: Specific inputs used.
        *   **Steps**: Actionable steps.
        *   **Expected Results**: Success criteria.
        *   **Negative & Boundary Cases**: Robustness checks.

---
