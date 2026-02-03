import fix_sqlite
import os
import hashlib
import json
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from processor import clean_and_serialize
from config import PERSIST_DIRECTORY, HASH_FILE, EMBEDDING_MODEL

def get_file_hash(file_bytes):
    return hashlib.md5(file_bytes).hexdigest()
def is_already_ingested(current_hash):
    from config import get_dataset_registry
    registry = get_dataset_registry()
    for dataset in registry["datasets"]:
        if dataset["hash"] == current_hash:
            return True
    return False

def save_dataset_to_registry(current_hash, db_path, filename, df):
    from config import get_dataset_registry
    os.makedirs(os.path.dirname(HASH_FILE), exist_ok=True)
    
    registry = get_dataset_registry()
    
    # Generate a unique path for the CSV to prevent overwrite
    import time
    csv_filename = f"data_{int(time.time())}_{filename.replace(' ', '_')}"
    if not csv_filename.endswith(".csv"):
        csv_filename += ".csv"
    csv_path = os.path.join(os.path.dirname(HASH_FILE), csv_filename)
    
    # Save CSV
    df.to_csv(csv_path, index=False)
    
    # Add to registry
    new_entry = {
        "filename": filename,
        "db_path": db_path,
        "csv_path": csv_path,
        "hash": current_hash
    }
    
    # Remove existing entry if it's the same hash (update scenario)
    registry["datasets"] = [d for dataset in [registry["datasets"]] for d in dataset if d["hash"] != current_hash]
    registry["datasets"].append(new_entry)
    
    with open(HASH_FILE, "w") as f:
        json.dump(registry, f)
    
    return new_entry

def ingest_dataset(uploaded_file, file_bytes):
    current_hash = get_file_hash(file_bytes)

    if is_already_ingested(current_hash):
        return "EXISTING"

    # 1. Clean and Serialize
    sentences, metadatas, df = clean_and_serialize(uploaded_file)

    if not sentences:
        return "ERROR: No readable data found."

    # 2. HuggingFace Embeddings (Local & Free)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # 3. ABSOLUTE Resilience: Create and Persist Vector DB with Error Recovery
    max_attempts = 3
    import time
    import uuid
    import shutil
    
    # Start with a fresh unique path for every ingestion to ensure isolation
    unique_id = str(uuid.uuid4())[:8]
    path_to_use = f"{PERSIST_DIRECTORY}_{int(time.time())}_{unique_id}"
    
    for attempt in range(max_attempts):
        try:
            if os.path.exists(path_to_use):
                shutil.rmtree(path_to_use, ignore_errors=True)
            os.makedirs(path_to_use, exist_ok=True)
            
            import chromadb
            client = chromadb.PersistentClient(
                path=path_to_use,
                settings=chromadb.config.Settings(
                    anonymized_telemetry=False,
                    is_persistent=True
                )
            )
            
            vectorstore = Chroma.from_texts(
                texts=sentences,
                metadatas=metadatas,
                embedding=embeddings,
                persist_directory=path_to_use,
                collection_name="employee_kb",
                client=client
            )
            
            save_dataset_to_registry(current_hash, path_to_use, uploaded_file.name, df)
            return "NEW"
            
        except Exception as e:
            err_msg = str(e).lower()
            # If "tenants" or "no such table" or "readonly" occurs, try a completely new path
            if any(x in err_msg for x in ["tenants", "readonly", "1032", "permission", "code: 1"]):
                if attempt < max_attempts - 1:
                    time.sleep(1) # Small cool-down
                    unique_id = str(uuid.uuid4())[:8]
                    path_to_use = f"{PERSIST_DIRECTORY}_{int(time.time())}_{unique_id}"
                    continue 
                else:
                    raise e
            else:
                raise e
    
    return "ERROR"