# 💼 Employee Intelligence Assistant
### AI-Powered Workforce Analytics with RAG

## 📋 Permissions & Requirements
*   **Python 3.10+**: Ensure Python is installed and accessible in your system path.
*   **Groq API Key**: A valid API key from [console.groq.com](https://console.groq.com/) is required for LLM inference.
*   **File System Permissions**: Read/write access is required for the `db/` and `metadata/` directories to persist data.
*   **Network Access**: Internet connection is required for initial model downloads and API communication with Groq.

## 📖 Overview
*   **Professional RAG Platform**: Built specifically for HR teams, Project Managers, and leadership.
*   **Interactive Intelligence**: Transforms raw employee spreadsheets into a data-grounded knowledge base.
*   **Advanced Models**: Leverages Llama 3.3 70B and ChromaDB for high-accuracy insights.
*   **Fact-Grounded**: Responses are strictly derived from retrieved data to eliminate hallucinations.

## 🚀 Key Features
*   **🧠 Semantic Workforce Understanding**:
    *   Converts CSV/Excel employee data into rich natural-language context.
    *   Optimized for structured HR and project-management datasets.
*   **💬 AI-Driven Query Interface**:
    *   Ask complex questions such as: "Who is overloaded this month?" or "Which roles are underutilized?"
    *   Compare workload across departments with accurate data retrieval.
*   **📊 Executive-Ready Dashboard**:
    *   Automated workforce metrics for quick leadership reviews.
    *   Interactive Plotly visualizations for trends, allocation, and utilization.
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
*   **`app.py`**: Streamlit UI, chat interface, and executive dashboards.
*   **`ingest.py`**: Dataset hashing, embedding generation, and vector DB persistence.
*   **`processor.py`**: Semantic cleaning and information-dense row chunking.
*   **`rag_engine.py`**: LangChain LCEL-based RAG pipeline construction.
*   **`fix_sqlite.py`**: Critical compatibility layer for Linux SQLite versions.
*   **`config.py`**: Centralized configuration for models, paths, and parameters.
*   **`db/`**: Persistent vector database (Excluded from Git).
*   **`metadata/`**: Dataset fingerprints and cached visuals (Excluded from Git).
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
*   **🔒 Local Isolation**: No employee data is sent to third-party training services.
*   **🧾 Git Protection**: `.gitignore` prevents committing sensitive `db/`, `metadata/`, and `.env` files.
*   **🧬 Data Integrity**: Each dataset creates a fresh, isolated intelligence layer.
*   **🧠 Collision Prevention**: Deterministic hashing ensures datasets don't overlap or conflict.

## 🎯 Ideal Use Cases
*   **HR Workload Analysis**: Identifying burnout risks or resource imbalances.
*   **Capacity Planning**: Evaluating team bandwidth for new project assignments.
*   **Utilization Insights**: Spotting underutilized roles or over-allocated leads.
*   **People Analytics**: Building internal intelligence tools without high cloud costs.
*   **RAG Demonstrations**: Showcasing production-grade retrieval logic for tabular data.
