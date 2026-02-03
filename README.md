# 💼 Employee Intelligence Assistant
### Enterprise-Grade Isolated Multi-Dataset RAG Architecture

## 📋 Permissions & Requirements
*   **Python 3.10+**: Ensure Python is installed and accessible in your system path.
*   **Groq API Key**: A valid API key from [console.groq.com](https://console.groq.com/) is required for LLM inference.
*   **File System Permissions**: Read/write access is required for the `db/` and `metadata/` directories to persist data.
*   **Network Access**: Internet connection is required for initial model downloads and API communication with Groq.

## 📖 Overview
*   **Isolated Intelligence Layers**: A sophisticated RAG platform that allows managing multiple datasets as distinct, isolated intelligence layers.
*   **Dataset Registry**: Built-in library management to switch between different team reports or project spreadsheets instantly.
*   **Advanced Models**: Leverages Llama 3.3 70B and ChromaDB Persistent Client for high-accuracy insights.
*   **Fact-Grounded**: Responses are strictly derived from retrieved data to eliminate hallucinations.

## 🚀 Key Features
*   **🧠 Multi-Dataset Registry**:
    *   Manage a library of uploaded files and switch context instantly.
    *   Per-dataset chat history ensures conversations remain isolated.
*   **💬 AI-Driven Contextual Querying**:
    *   Ask complex questions targeted at specific datasets.
    *   Automatic "Pointer Reset" ensures reliable ingestion for CSV and Excel files.
*   **📊 Executive-Ready Dashboard**:
    *   Dynamic metrics and Plotly visualizations that refresh based on the active dataset.
*   **🔐 Atomic Data Governance**:
    *   **Individual Deletion**: Remove specific datasets with a single click (clears DB and metadata).
    *   **Factory Reset**: Complete system purge with a safety confirmation dialog.
*   **🔐 Privacy-First Design**:
    *   Uses local embeddings (`sentence-transformers`) so data processing stays local.
    *   Employee data never leaves your machine; only inference queries reach the Groq API.
*   **🖥️ Cross-Platform Support**:
    *   Fully compatible with Windows, Linux, and macOS.
    *   Includes a dedicated SQLite compatibility fix for Linux environments.

## 🛠️ Tech Stack
*   **Frontend**: Streamlit
*   **LLM**: Llama 3.3 70B (via Groq API)
*   **Vector Database**: ChromaDB
*   **RAG Framework**: LangChain (LCEL)
*   **Embeddings**: sentence-transformers (Local)
*   **Data & Visualization**: Pandas, Plotly

## 📥 Installation

### 1️⃣ Clone & Environment Setup
*   **Clone Repository**: `git clone <your-repo-url>`
*   **Navigate**: `cd Project_Employee_AI_Assistent`
*   **Create Virtual Environment**: `python -m venv venv`
*   **Activate Environment (Linux/macOS)**: `source venv/bin/activate`
*   **Activate Environment (Windows)**: `venv\Scripts\activate`
*   **Install Dependencies**: `pip install -r requirements.txt`

### 2️⃣ Environment Configuration
*   **Create .env File**: Create a file named `.env` in the project root.
*   **Configure API Key**: Add the following content to the file:
    ```env
    # Groq API Key (Required)
    # Get yours here: https://console.groq.com/keys
    GROQ_API_KEY=your_api_key_here
    ```

## 🏃 Running the Application
*   **Start Server**: `streamlit run app.py`
*   **Access Dashboard**: Open `http://localhost:8501` in your browser.

## 📂 Project Structure
*   **`app.py`**: Multi-dataset UI, context-aware chat, and dynamic dashboard.
*   **`ingest.py`**: Registry management, isolated embedding generation, and UUID-based persistence.
*   **`app_config.py`**: Dataset Registry logic and centralized model configurations.
*   **`rag_engine.py`**: Context-aware RAG pipeline supporting dynamic DB connections.
*   **`processor.py`**: Semantic cleaning and robust pointer-reset ingestion.
*   **`fix_sqlite.py`**: Critical compatibility layer for Linux SQLite versions.
*   **`db/`**: Persistent vector databases (Excluded from Git).
*   **`metadata/`**: Dataset Registry file and isolated CSV copies (Excluded from Git).
*   **`.env`**: Private environment variables (Excluded from Git).

## 🧠 How the RAG Pipeline Works
1.  **Upload**: User uploads an employee dataset (CSV / Excel).
2.  **Normalize**: The system performs semantic normalization of structured rows.
3.  **Chunk**: Data is converted into information-dense natural language phrases.
4.  **Embed**: Local models generate mathematical vectors for each chunk.
5.  **Store**: Vectors are persisted in the local ChromaDB.
6.  **Retrieve**: Relevant context is pulled based on the user's semantic query.
7.  **Generate**: LLM provides a factual answer grounded strictly in the retrieved data.

## 🛡️ Privacy & Security
*   **🔒 Isolated Contexts**: Data from "Dataset A" never bleeds into "Dataset B."
*   **🧾 Zero-Trace Deletion**: Deleting a dataset performs an atomic purge of all related disk artifacts.
*   **🧠 Privacy-First**: Local embeddings ensure employee data stays within your controlled environment.

## 🎯 Ideal Use Cases
*   **HR Workload Analysis**: Identifying burnout risks or resource imbalances.
*   **Capacity Planning**: Evaluating team bandwidth for new project assignments.
*   **Utilization Insights**: Spotting underutilized roles or over-allocated leads.
*   **People Analytics**: Building internal intelligence tools without high cloud costs.
*   **RAG Demonstrations**: Showcasing production-grade retrieval logic for tabular data.
