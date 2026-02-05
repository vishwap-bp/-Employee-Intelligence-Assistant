import pandas as pd
from typing import List, Tuple

def clean_and_serialize(uploaded_file) -> Tuple[List[str], List[dict], pd.DataFrame]:
    """
    OPTIMIZED Processor: Creates clean, information-dense chunks for accurate RAG.
    Handles Clockify data format specifically to prevent data corruption.
    """
    # 1. Load with robust encoding fallbacks
    try:
        uploaded_file.seek(0) # Ensure we start from the beginning
        if uploaded_file.name.endswith(".csv"):
            # Try with different parameters for better CSV handling
            df = pd.read_csv(uploaded_file, low_memory=False)
        else:
            df = pd.read_excel(uploaded_file)
    except UnicodeDecodeError:
        uploaded_file.seek(0) # Reset for the fallback attempt
        df = pd.read_csv(uploaded_file, encoding='latin1', low_memory=False)
    except Exception as e:
        # Handle other potential loading errors
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, encoding='utf-8', low_memory=False)

    # 2. Advanced Cleaning
    df = df.dropna(how="all").dropna(axis=1, how="all")
    
    # Store original data info for debugging
    original_shape = df.shape
    original_columns = df.columns.tolist()
    
    # Standardize column names
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    
    # Debug information (can be removed in production)
    print(f"Original data shape: {original_shape}")
    print(f"Columns: {original_columns}")
    print(f"Standardized columns: {df.columns.tolist()}")
    
    # Check for potential data corruption issues
    numeric_cols = df.select_dtypes(include=['number']).columns
    print(f"Numeric columns: {list(numeric_cols)}")
    for col in numeric_cols:
        zero_count = (df[col] == 0).sum()
        nan_count = df[col].isna().sum()
        print(f"Column '{col}': {zero_count} zeros, {nan_count} NaNs")
    
    # Additional debugging for Clockify data
    date_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['date', 'time'])]
    print(f"Date/Time columns identified: {date_cols}")
    for col in date_cols:
        sample_values = df[col].dropna().head(3).tolist()
        print(f"Sample values from '{col}': {sample_values}")

    # Identify key columns
    col_map = {
        "person": next((c for c in df.columns if any(x in c for x in ['name', 'employee', 'consultant', 'person'])), None),
        "project": next((c for c in df.columns if any(x in c for x in ['project', 'task', 'client'])), None),
        "date": next((c for c in df.columns if any(x in c for x in ['date', 'time', 'period'])), None)
    }

    # Normalize data - More precise column handling
    for col in df.columns:
        # Handle date columns specifically (avoid affecting numeric time columns like Hours)
        if any(keyword in col.lower() for keyword in ['date', 'start_date', 'end_date', 'created', 'modified']):
            # Only convert actual date columns, not time duration columns
            if not any(time_keyword in col.lower() for time_keyword in ['hours', 'minutes', 'duration', 'time_spent', 'decimal']):
                # Try multiple date formats to handle Clockify data properly
                try:
                    # Try common date formats
                    date_formats = ['%m/%d/%Y', '%d/%m/%Y', '%Y-%m-%d', '%m-%d-%Y']
                    parsed_date = None
                    for fmt in date_formats:
                        try:
                            parsed_date = pd.to_datetime(df[col], format=fmt, errors='raise')
                            break
                        except (ValueError, TypeError):
                            continue
                    
                    # If no specific format works, let pandas infer
                    if parsed_date is None:
                        parsed_date = pd.to_datetime(df[col], errors="coerce")
                        
                    df[col] = parsed_date.dt.strftime('%B %d, %Y')
                except Exception:
                    # If all parsing fails, keep original value
                    pass
        elif pd.api.types.is_numeric_dtype(df[col]):
            # Only fill numeric columns that are actually NaN, preserve 0 values
            df[col] = df[col].fillna(0) if df[col].isna().any() else df[col]
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