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
# Base directories
BASE_VECTOR_DB_DIR = os.path.abspath("./db")
BASE_METADATA_DIR = os.path.abspath("./metadata")

# User-specific paths will be determined dynamically
PERSIST_DIRECTORY = None  # Will be set per user
METADATA_DIR = None       # Will be set per user
HASH_FILE = None          # Will be set per user


def get_user_storage_paths(username):
    """Generate user-specific storage paths"""
    user_safe = username.replace(" ", "_").replace("@", "_").replace(".", "_")
    user_db_dir = os.path.join(BASE_VECTOR_DB_DIR, user_safe)
    user_metadata_dir = os.path.join(BASE_METADATA_DIR, user_safe)
    user_hash_file = os.path.join(user_metadata_dir, "dataset_hash.json")
    
    # Create directories if they don't exist
    os.makedirs(user_db_dir, exist_ok=True)
    os.makedirs(user_metadata_dir, exist_ok=True)
    
    return {
        "vector_db": user_db_dir,
        "metadata": user_metadata_dir,
        "hash_file": user_hash_file
    }

def get_dataset_registry(username=None):
    """
    Retrieves the full registry of uploaded datasets for a specific user.
    Structure: {"datasets": [{"filename": str, "db_path": str, "csv_path": str, "hash": str}]}
    """
    # Use default user if none specified (for backward compatibility)
    if username is None:
        username = "default_user"
    
    user_paths = get_user_storage_paths(username)
    hash_file = user_paths["hash_file"]
    
    if os.path.exists(hash_file):
        try:
            with open(hash_file, "r") as f:
                data = json.load(f)
                if "datasets" in data:
                    return data
                # Backward compatibility for old single-file format
                if "active_db_path" in data:
                    return {"datasets": [{
                        "filename": data.get("filenames", ["Legacy Dataset"])[0],
                        "db_path": data.get("active_db_path"),
                        "csv_path": os.path.join(user_paths["metadata"], "active_data.csv"),
                        "hash": data.get("hashes", ["unknown"])[0]
                    }]}
        except:
            pass
    return {"datasets": []}

def get_active_db_path(username=None):
    """
    Backward compatibility helper that works with user-specific data.
    """
    registry = get_dataset_registry(username)
    if registry["datasets"]:
        return registry["datasets"][-1]["db_path"]
    return PERSIST_DIRECTORY
