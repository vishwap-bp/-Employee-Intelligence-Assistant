# app_config.py
import os
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# =========================
# API CONFIGURATION
# =========================
# Ensure your .env file has GROQ_API_KEY=your_key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# =========================
# MODEL SELECTION
# =========================
# Llama 3 70B via Groq (Free, Fast, High Intelligence)
LLM_MODEL = "llama-3.3-70b-versatile"  # Updated: llama3-70b-8192 deprecated
# Local Embeddings (Free, Persistent, No API limits)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# =========================
# RAG STRATEGY
# =========================
# 0.0 is the industry standard for factual data like employee records
TEMPERATURE = 0.0        
# k=10 allows the AI to see enough rows to compare workloads effectively
TOP_K = 10

# STORAGE PATHS
# =========================
VECTOR_DB_DIR = os.path.abspath("./db")
PERSIST_DIRECTORY = os.path.join(VECTOR_DB_DIR, "chroma")
METADATA_DIR = os.path.abspath("./metadata")
HASH_FILE = os.path.join(METADATA_DIR, "dataset_hash.json")

def get_dataset_registry():
    """
    Retrieves the full registry of uploaded datasets.
    Structure: {"datasets": [{"filename": str, "db_path": str, "csv_path": str, "hash": str}]}
    """
    if os.path.exists(HASH_FILE):
        try:
            with open(HASH_FILE, "r") as f:
                data = json.load(f)
                if "datasets" in data:
                    return data
                # Backward compatibility for old single-file format
                if "active_db_path" in data:
                    return {"datasets": [{
                        "filename": data.get("filenames", ["Legacy Dataset"])[0],
                        "db_path": data.get("active_db_path"),
                        "csv_path": os.path.join(METADATA_DIR, "active_data.csv"),
                        "hash": data.get("hashes", ["unknown"])[0]
                    }]}
        except:
            pass
    return {"datasets": []}

def get_active_db_path():
    """
    Backward compatibility helper.
    """
    registry = get_dataset_registry()
    if registry["datasets"]:
        return registry["datasets"][-1]["db_path"]
    return PERSIST_DIRECTORY
