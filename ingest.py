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
    if not os.path.exists(HASH_FILE):
        return False
    try:
        with open(HASH_FILE, "r") as f:
            content = f.read().strip()
            if not content: return False
            saved = json.loads(content)
            # Support both single 'hash' and list of 'hashes'
            hashes = saved.get("hashes", [])
            if not isinstance(hashes, list):
                hashes = [saved.get("hash")] if "hash" in saved else []
            return current_hash in hashes
    except (json.JSONDecodeError, ValueError):
        return False

def save_dataset_hash(current_hash, path_used, filename, df=None):
    os.makedirs(os.path.dirname(HASH_FILE), exist_ok=True)
    
    hashes = []
    filenames = []
    if os.path.exists(HASH_FILE):
        try:
            with open(HASH_FILE, "r") as f:
                data = json.load(f)
                hashes = data.get("hashes", [])
                filenames = data.get("filenames", [])
                if not isinstance(hashes, list):
                    hashes = [data.get("hash")] if "hash" in data else []
        except:
            hashes = []
            filenames = []

    if current_hash not in hashes:
        hashes.append(current_hash)
    
    if filename and filename not in filenames:
        filenames.append(filename)
            
    # Save the dataframe for persistent dashboard access
    if df is not None:
        df_path = os.path.join(os.path.dirname(HASH_FILE), "active_data.csv")
        df.to_csv(df_path, index=False)
            
    with open(HASH_FILE, "w") as f:
        json.dump({
            "hashes": hashes,
            "filenames": filenames,
            "active_db_path": path_used
        }, f)

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
    max_attempts = 2
    path_to_use = PERSIST_DIRECTORY
    
    for attempt in range(max_attempts):
        try:
            os.makedirs(path_to_use, exist_ok=True)
            
            vectorstore = Chroma.from_texts(
                texts=sentences,
                metadatas=metadatas,
                embedding=embeddings,
                persist_directory=path_to_use,
                collection_name="employee_kb"
            )
            save_dataset_hash(current_hash, path_to_use, uploaded_file.name, df)
            return "NEW"
            
        except Exception as e:
            err_msg = str(e).lower()
            if "readonly" in err_msg or "1032" in err_msg:
                if attempt < max_attempts - 1:
                    import hashlib
                    path_to_use = f"{PERSIST_DIRECTORY}_{int(hashlib.md5(str(e).encode()).hexdigest()[:8])}"
                    continue # Try again with unique path
                else:
                    raise e
            elif "tenants" in err_msg or "code: 1" in err_msg:
                if attempt < max_attempts - 1:
                    # Critical schema error - Wipe and retry fresh
                    import shutil
                    if os.path.exists(path_to_use):
                        shutil.rmtree(path_to_use, ignore_errors=True)
                    continue 
                else:
                    raise e
            else:
                raise e
    
    return "ERROR"