import pandas as pd
from typing import List, Tuple

def clean_and_serialize(uploaded_file) -> Tuple[List[str], List[dict], pd.DataFrame]:
    """
    OPTIMIZED Processor: Creates clean, information-dense chunks for accurate RAG.
    """
    # 1. Load with robust encoding fallbacks
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except UnicodeDecodeError:
        df = pd.read_csv(uploaded_file, encoding='latin1')

    # 2. Advanced Cleaning
    df = df.dropna(how="all").dropna(axis=1, how="all")
    
    # Standardize column names
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    # Identify key columns
    col_map = {
        "person": next((c for c in df.columns if any(x in c for x in ['name', 'employee', 'consultant', 'person'])), None),
        "project": next((c for c in df.columns if any(x in c for x in ['project', 'task', 'client'])), None),
        "date": next((c for c in df.columns if any(x in c for x in ['date', 'time', 'period'])), None)
    }

    # Normalize data
    for col in df.columns:
        if "date" in col or "time" in col:
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.strftime('%B %d, %Y')
        elif pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(0)
        else:
            df[col] = df[col].fillna("Not Specified")

    # 3. IMPROVED: Create Natural, Information-Dense Chunks
    sentences = []
    metadatas = []
    
    for idx, row in df.iterrows():
        # Build natural sentence parts
        parts = []
        person_val = "Unknown"
        project_val = "General"
        date_val = "Ongoing"
        
        for col, val in row.items():
            # Skip empty or redundant values
            if val in ["Not Specified", 0, "0", ""]:
                continue
                
            # Extract metadata
            if col_map["person"] and col == col_map["person"]:
                person_val = str(val)
                continue  # Don't repeat in sentence
            if col_map["project"] and col == col_map["project"]:
                project_val = str(val)
                continue
            if col_map["date"] and col == col_map["date"]:
                date_val = str(val)
                continue
            
            # Create natural phrases
            clean_col = col.replace("_", " ").title()
            parts.append(f"{clean_col}: {val}")
        
        # Construct NATURAL sentence
        if parts:
            sentence = f"{person_val} ({project_val}): {'; '.join(parts)}"
        else:
            sentence = f"{person_val} worked on {project_val}"
        
        sentences.append(sentence)
        metadatas.append({
            "person": person_val,
            "project": project_val,
            "date": date_val,
            "row_index": idx
        })

    return sentences, metadatas, df