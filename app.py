import fix_sqlite # Must be first
import os
import json
import shutil
import time
import gc
import hashlib
import streamlit as st
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv
from streamlit_extras.metric_cards import style_metric_cards

# Importing your logic modules
from processor import clean_and_serialize
from rag_engine import get_rag_chain
from app_config import PERSIST_DIRECTORY, get_dataset_registry

# 1. Environment & Security
load_dotenv()
api_key = os.getenv("GROQ_API_KEY") 

# 2. Page Configuration
st.set_page_config(
    page_title="Employee Intelligence Assistant",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 3. COHESIVE DARK THEME - NO WHITE ANYWHERE
st.markdown("""
    <style>
    /* ========== UNIFIED DARK THEME ========== */
    :root {
        --bg-primary: #0F1419;
        --bg-secondary: #1A1F2E;
        --bg-tertiary: #242B3D;
        --bg-card: #1E2433;
        --primary-blue: #4A9EFF;
        --primary-blue-light: #6BB1FF;
        --primary-blue-dark: #357ABD;
        --text-primary: #E8EAED;
        --text-secondary: #9BA3B0;
        --text-muted: #6B7280;
        --border-color: #2D3748;
        --border-light: #374151;
        --success-green: #10B981;
        --danger-red: #EF4444;
        --warning-orange: #F59E0B;
    }
    
    /* ========== MAIN BACKGROUND ========== */
    .main {
        background: linear-gradient(135deg, var(--bg-primary) 0%, var(--bg-secondary) 100%) !important;
    }
    
    /* ========== SIDEBAR - MATCHING DARK THEME ========== */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--bg-secondary) 0%, var(--bg-tertiary) 100%) !important;
        border-right: 1px solid var(--border-color) !important;
    }
    
    [data-testid="stSidebar"] .stMarkdown h1 {
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        color: var(--primary-blue) !important;
        margin-bottom: 1rem !important;
        padding-bottom: 0.5rem !important;
        border-bottom: 2px solid var(--primary-blue);
    }
    
    [data-testid="stSidebar"] .stMarkdown h3 {
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        color: var(--text-secondary) !important;
        margin-top: 1.2rem !important;
        margin-bottom: 0.6rem !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    [data-testid="stSidebar"] .stMarkdown h4 {
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        color: var(--text-secondary) !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Sidebar text colors */
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stFileUploader label,
    [data-testid="stSidebar"] label {
        color: var(--text-primary) !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
    }
    
    /* ========== CHAT UI - DARK BUBBLES ========== */
    
    /* Chat messages container */
    .stChatMessage {
        padding: 1rem !important;
        margin-bottom: 0.75rem !important;
        border-radius: 16px !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3) !important;
        max-width: 85% !important;
        transition: all 0.2s ease !important;
    }
    
    .stChatMessage:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.4) !important;
    }
    
    /* User messages (right side - blue gradient) */
    [data-testid="stChatMessageContainer"] div:has(> .stChatMessage[data-testid*="user"]) {
        background: linear-gradient(135deg, #4A9EFF 0%, #357ABD 100%) !important;
        color: white !important;
        border-radius: 18px 18px 4px 18px !important;
        margin-left: auto !important;
    }
    
    /* Assistant messages (left side - dark card) */
    [data-testid="stChatMessageContainer"] div:has(> .stChatMessage[data-testid*="assistant"]) {
        background: var(--bg-card) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 18px 18px 18px 4px !important;
        margin-right: auto !important;
    }
    
    /* ────────────────────────────────────────────────
   Clean single blue border chat input + blue arrow
──────────────────────────────────────────────── */

.stChatInput {
    /* Main container */
    background: var(--bg-tertiary) !important;
    border-radius: 24px !important;
    border: 2px solid var(--primary-blue) !important;
    padding: 4px 8px !important;           /* reduced padding */
    box-shadow: none !important;
    outline: none !important;
}

/* Remove any inner borders from the text input itself */
.stChatInput > div > div > textarea,
.stChatInput input,
.stChatInput textarea {
    background: transparent !important;
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
    color: var(--text-primary) !important;
    padding: 12px 16px !important;
    font-size: 1rem !important;
}

/* Send button / icon area */
.stChatInput button[kind="primary"] {
    background: transparent !important;
    border: none !important;
    color: var(--primary-blue) !important;
    padding: 0 12px !important;
    margin-right: 4px !important;
}

.stChatInput button[kind="primary"] svg {
    fill: var(--primary-blue) !important;
    stroke: var(--primary-blue) !important;
    width: 24px !important;
    height: 24px !important;
}

/* Hover & focus states — keep single border, just brighter */
.stChatInput:hover,
.stChatInput:focus-within {
    border-color: var(--primary-blue-light) !important;
    box-shadow: 0 0 0 3px rgba(74, 158, 255, 0.15) !important;
}

/* Remove any focus ring that might come from browser defaults */
.stChatInput:focus-within {
    outline: none !important;
}

/* Optional: make placeholder nicer */
.stChatInput textarea::placeholder {
    color: var(--text-muted) !important;
    opacity: 0.7 !important;
}
    
 .stChatFloatingInputContainer {
    background: var(--bg-secondary) !important;
    border-top: 1px solid var(--border-color) !important;
    padding: 16px 16px 24px !important;  /* give some breathing room */
}
    
    /* ========== BUTTONS - DARK THEME ========== */
    .stButton > button {
        border-radius: 10px !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
        border: none !important;
        font-size: 0.9rem !important;
        background: var(--bg-card) !important;
        color: var(--text-primary) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.4) !important;
    }
    
    /* Primary button */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, var(--primary-blue) 0%, var(--primary-blue-dark) 100%) !important;
        color: white !important;
    }
    
    /* Secondary button */
    .stButton > button[kind="secondary"] {
        background: var(--bg-card) !important;
        border: 2px solid var(--danger-red) !important;
        color: var(--danger-red) !important;
    }
    
    .stButton > button[kind="secondary"]:hover {
        background: var(--danger-red) !important;
        color: white !important;
    }
    
    /* ========== TABS - DARK STYLING ========== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
        border-bottom: 2px solid var(--border-color);
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        font-size: 0.95rem;
        color: var(--text-secondary);
        background: transparent;
        border: none;
        transition: all 0.2s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--bg-card);
        color: var(--primary-blue);
        border-bottom: 3px solid var(--primary-blue) !important;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(74, 158, 255, 0.1);
    }
    
    /* ========== METRICS CARDS - DARK ========== */
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 700 !important;
        color: var(--primary-blue) !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.9rem !important;
        font-weight: 500 !important;
        color: var(--text-secondary) !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* ========== DIVIDERS ========== */
    hr {
        margin: 1.5rem 0 !important;
        border: none !important;
        border-top: 1px solid var(--border-color) !important;
    }
    
    /* ========== STATUS & ALERTS - DARK ========== */
    .stSuccess {
        background: rgba(16, 185, 129, 0.15) !important;
        border-left: 4px solid var(--success-green) !important;
        border-radius: 8px !important;
        padding: 1rem !important;
        color: var(--text-primary) !important;
    }
    
    .stError {
        background: rgba(239, 68, 68, 0.15) !important;
        border-left: 4px solid var(--danger-red) !important;
        border-radius: 8px !important;
        padding: 1rem !important;
        color: var(--text-primary) !important;
    }
    
    .stWarning {
        background: rgba(245, 158, 11, 0.15) !important;
        border-left: 4px solid var(--warning-orange) !important;
        border-radius: 8px !important;
        padding: 1rem !important;
        color: var(--text-primary) !important;
    }
    
    .stInfo {
        background: rgba(74, 158, 255, 0.15) !important;
        border-left: 4px solid var(--primary-blue) !important;
        border-radius: 8px !important;
        padding: 1rem !important;
        color: var(--text-primary) !important;
    }
    
    /* ========== TITLES & HEADERS - LIGHT TEXT ========== */
    h1, h2, h3, h4, h5, h6 {
        color: var(--text-primary) !important;
        font-weight: 600 !important;
    }
    
    h1 {
        font-weight: 700 !important;
        letter-spacing: -0.5px !important;
    }
    
    /* ========== FILE UPLOADER - DARK ========== */
    [data-testid="stFileUploader"] {
        border: 2px dashed var(--border-light) !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
        background: var(--bg-card) !important;
        transition: all 0.3s ease !important;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: var(--primary-blue) !important;
        background: rgba(74, 158, 255, 0.1) !important;
    }
    
    [data-testid="stFileUploader"] section {
        border: none !important;
    }
    
    /* ========== SELECTBOX - DARK ========== */
    .stSelectbox > div > div {
        border-radius: 8px !important;
        border: 1px solid var(--border-color) !important;
        background: var(--bg-card) !important;
        color: var(--text-primary) !important;
        transition: all 0.2s ease !important;
    }
    
    .stSelectbox > div > div:focus-within {
        border-color: var(--primary-blue) !important;
        box-shadow: 0 0 0 3px rgba(74, 158, 255, 0.2) !important;
    }
    
    /* ========== DATAFRAME - DARK ========== */
    .stDataFrame {
        border: 1px solid var(--border-color) !important;
        border-radius: 12px !important;
        overflow: hidden !important;
    }
    
    /* ========== PLOTLY CHARTS - DARK BACKGROUND ========== */
    .js-plotly-plot {
        border-radius: 12px !important;
        overflow: hidden !important;
        background: var(--bg-card) !important;
    }
    
    /* ========== ANIMATIONS ========== */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .stChatMessage {
        animation: fadeIn 0.3s ease-out;
    }
    
    /* ========== SPINNER ========== */
    .stSpinner > div {
        border-color: var(--primary-blue) !important;
    }
    
    /* ========== CAPTIONS ========== */
    .stCaption {
        color: var(--text-secondary) !important;
        font-size: 0.85rem !important;
    }
    
    /* ========== TEXT INPUTS - DARK ========== */
    input, textarea {
        background: var(--bg-card) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-color) !important;
    }
    
    input:focus, textarea:focus {
        border-color: var(--primary-blue) !important;
        box-shadow: 0 0 0 3px rgba(74, 158, 255, 0.2) !important;
    }
    
    /* ========== MARKDOWN CONTENT ========== */
    .stMarkdown {
        color: var(--text-primary) !important;
    }
    
    </style>
    """, unsafe_allow_html=True)

# 2.5 Session State Initialization
# Initialize user identification
if "username" not in st.session_state:
    import random
    import string
    # Generate a random username for anonymous users
    # Create a session-unique identifier using timestamp and randomness
    import time
    timestamp_part = str(int(time.time()))[-6:]  # Last 6 digits of timestamp
    random_part = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
    st.session_state.username = f"user_{timestamp_part}{random_part}"

if "registry" not in st.session_state:
    st.session_state.registry = get_dataset_registry(st.session_state.username)

if "active_dataset" not in st.session_state:
    # Default to the most recent dataset if available
    if st.session_state.registry["datasets"]:
        st.session_state.active_dataset = st.session_state.registry["datasets"][-1]
    else:
        st.session_state.active_dataset = None

if "chat_histories" not in st.session_state:
    st.session_state.chat_histories = {} # keyed by dataset hash

# Helper to refresh registry
def refresh_registry():
    st.session_state.registry = get_dataset_registry(st.session_state.username)

# 3. Sidebar Controls
with st.sidebar:
    st.title("💼 Employee AI")
    st.markdown("### 📤 Upload New Data")
    
    uploaded_file = st.file_uploader(
        "Upload Team Dataset",
        type=["csv", "xlsx"],
        help="Upload a new employee spreadsheet to build a dedicated intelligence layer."
    )
    
    # File size validation
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    if uploaded_file and uploaded_file.size > MAX_FILE_SIZE:
        st.error(f"❌ File too large ({uploaded_file.size / (1024*1024):.1f}MB). Please upload files under 50MB.")
        uploaded_file = None
    
    if uploaded_file:
        # Check if already processed in this session to avoid loop
        if "last_uploaded" not in st.session_state or st.session_state.last_uploaded != uploaded_file.name:
            try:
                with st.status("🛠️ Building Intelligence Layer...", expanded=True) as status:
                    # Add detailed progress information
                    status.update(label="📥 Loading file...", state="running")
                    bytes_data = uploaded_file.read()
                    uploaded_file.seek(0) # Reset pointer for Pandas
                    status.update(label="🔄 Processing data...", state="running")
                    from ingest import ingest_dataset
                    status_code = ingest_dataset(uploaded_file, bytes_data, st.session_state.username)
                    
                    if status_code in ["NEW", "EXISTING"]:
                        status.update(label="📊 Updating registry...", state="running")
                        refresh_registry()
                        # Auto-select the newly uploaded file
                        current_hash = hashlib.md5(bytes_data).hexdigest()
                        
                        for d in st.session_state.registry["datasets"]:
                            if d["hash"] == current_hash:
                                st.session_state.active_dataset = d
                                break
                        
                        st.session_state.last_uploaded = uploaded_file.name
                        status.update(label="✅ Intelligence Layer Ready!", state="complete", expanded=False)
                        st.success("✅ Dataset processed successfully!")
                        st.rerun()
                    else:
                        status.update(label="❌ Processing failed", state="error")
                        st.error(f"Dataset processing failed: {status_code}")
            except Exception as e:
                st.error(f"Upload failed: {e}")

    st.divider()
    
    # Dataset Library
    st.markdown("### 🗄️ Dataset Library")
    datasets = st.session_state.registry["datasets"]
    
    if not datasets:
        st.info("No datasets uploaded yet.")
    else:
        # Selection Dropdown
        options = [d["filename"] for d in datasets]
        current_index = 0
        if st.session_state.active_dataset:
            try:
                current_index = options.index(st.session_state.active_dataset["filename"])
            except: pass
            
        selected_name = st.selectbox(
            "Active Dataset",
            options=options,
            index=current_index,
            help="Select which dataset you want to analyze and chat with."
        )
        
        # Update active dataset based on selection
        for d in datasets:
            if d["filename"] == selected_name:
                if st.session_state.active_dataset != d:
                    st.session_state.active_dataset = d
                    st.rerun()
                break

        st.divider()
        st.markdown("#### Manage Files")
        for i, d in enumerate(datasets):
            col1, col2 = st.columns([4, 1])
            col1.caption(f"📄 {d['filename']}")
            if col2.button("🗑️", key=f"del_{i}", help=f"Delete {d['filename']}"):
                try:
                    # 1. Physical Delete
                    if os.path.exists(d["db_path"]):
                        shutil.rmtree(d["db_path"], ignore_errors=True)
                    if os.path.exists(d["csv_path"]):
                        os.remove(d["csv_path"])
                    
                    # 2. Registry Delete
                    from app_config import get_dataset_registry, get_user_storage_paths
                    user_paths = get_user_storage_paths(st.session_state.username)
                    new_datasets = [dataset for dataset in datasets if dataset["hash"] != d["hash"]]
                    with open(user_paths["hash_file"], "w") as f:
                        json.dump({"datasets": new_datasets}, f)
                    
                    # 3. State cleanup
                    if st.session_state.active_dataset and st.session_state.active_dataset["hash"] == d["hash"]:
                        st.session_state.active_dataset = None
                    
                    refresh_registry()
                    st.success(f"Deleted {d['filename']}")
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error(f"Could not delete: {e}")

    st.divider()
    # clear chat history for active dataset
    if st.session_state.get("active_dataset"):
        d = st.session_state.active_dataset
        history_key = d["hash"]
        if st.session_state.chat_histories.get(history_key):
            if st.button("🗑️ Clear Chat History", use_container_width=True):
                st.session_state.chat_histories[history_key] = []
                st.rerun()

    
    # Factory Reset
    if datasets:
        if st.button("🚨 Factory Reset", type="secondary", use_container_width=True):
            st.session_state.show_factory_confirm = True
            
        if st.session_state.get("show_factory_confirm", False):
            st.warning("Permanently delete ALL data?")
            c1, c2 = st.columns(2)
            if c1.button("✅ Yes", type="primary"):
                try:
                    import gc
                    
                    # Step 1: Clear Streamlit Resource Cache (Critical for releasing handles)
                    st.cache_resource.clear()
                    
                    # Step 2: Force reset of RAG objects
                    if 'rag_chain' in st.session_state:
                        st.session_state.rag_chain = None
                        del st.session_state.rag_chain
                    
                    # Step 3: Clear ALL dataset-related session state
                    if 'df' in st.session_state:
                        st.session_state.df = None
                        del st.session_state.df
                    
                    if 'active_dataset' in st.session_state:
                        del st.session_state.active_dataset
                    
                    if 'registry' in st.session_state:
                        del st.session_state.registry
                        
                    if 'chat_histories' in st.session_state:
                        del st.session_state.chat_histories
                    
                    # Clear upload session state
                    if 'last_uploaded' in st.session_state:
                        del st.session_state.last_uploaded
                    
                    # CRITICAL: Reset KB status
                    st.session_state.kb_status = "INACTIVE"
                    if "kb_files" in st.session_state: 
                        del st.session_state.kb_files
                    
                    # Step 4: Ultra-aggressive garbage collection
                    for _ in range(5):
                        gc.collect()
                    time.sleep(2.0)  # Wait for OS to catch up
                    
                    # Step 5: Try to delete with retry logic
                    max_retries = 3
                    from app_config import get_user_storage_paths
                    user_paths = get_user_storage_paths(st.session_state.username)
                    
                    db_parent = user_paths["vector_db"]
                    
                    for attempt in range(max_retries):
                        try:
                            # Delete all vector database folders for this user
                            if os.path.exists(db_parent):
                                for item in os.listdir(db_parent):
                                    item_path = os.path.join(db_parent, item)
                                    if os.path.isdir(item_path):
                                        shutil.rmtree(item_path)
                            break  # Success!
                        except (PermissionError, OSError) as e:
                            if attempt < max_retries - 1:
                                time.sleep(2.0) 
                                gc.collect()
                            else:
                                # Fallback: rename directory if it can't be deleted
                                if os.path.exists(db_parent):
                                    try:
                                        shutil.move(db_parent, f"{db_parent}_old_{int(time.time())}")
                                    except:
                                        raise e
                    
                    # Step 6: Delete all metadata files for this user (including registry)
                    metadata_parent = user_paths["metadata"]
                    if os.path.exists(metadata_parent):
                        # Delete all files including dataset_hash.json
                        for f in os.listdir(metadata_parent):
                            f_path = os.path.join(metadata_parent, f)
                            if os.path.isfile(f_path):
                                os.remove(f_path)
                        
                        # Delete the metadata directory itself to ensure clean slate
                        try:
                            shutil.rmtree(metadata_parent)
                        except:
                            pass  # If can't delete directory, at least files are gone
                    
                    # Step 7: Force complete refresh by clearing confirmation flag BEFORE rerun
                    st.session_state.show_factory_confirm = False
                    
                    # Show success message
                    st.success("✅ Knowledge base cleared!")
                    st.info("💡 **Ready for new data.** You can now upload a fresh file.")
                    
                    # CRITICAL: Rerun to refresh UI
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"⚠️ Could not fully clear KB: {e}")
                    st.warning("**Workaround**: Restart the application or refresh the browser.")
            if c2.button("❌ No"):
                st.session_state.show_factory_confirm = False
                st.rerun()

# 4. Main Interface Logic
if not api_key:
    st.error("🔑 Groq API Key missing. Add it to your .env file.")
    st.stop()

if st.session_state.active_dataset:
    d = st.session_state.active_dataset
    
    # Load Data for Dashboard
    try:
        df = pd.read_csv(d["csv_path"])
        st.session_state.df = df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

    st.title(f"📊 Analyzing: {d['filename']}")
    
    tab1, tab2, tab3 = st.tabs(["💬 AI Chat", "📈 Dashboard", "📂 Raw Data"])
    
    with tab1:
        history_key = d["hash"]
        if history_key not in st.session_state.chat_histories:
            st.session_state.chat_histories[history_key] = []

    # ---- Scrollable chat area ----
        chat_box = st.container(height=500)

        with chat_box:
            for msg in st.session_state.chat_histories[history_key]:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

    # ---- Chat input (STATIC position) ----
        prompt = st.chat_input(f"Ask me anything about {d['filename']}...", key="chat_input")

        if prompt:
            st.session_state.chat_histories[history_key].append(
                {"role": "user", "content": prompt}
            )

            with st.chat_message("assistant"):
                with st.spinner("Let me check that for you..."):
                    chain = get_rag_chain(
                        api_key,
                        db_path=d["db_path"],
                        username=st.session_state.username
                    )
                    # Pass chat history for conversational context
                    result = chain.invoke({
                        "input": prompt,
                        "chat_history": st.session_state.chat_histories[history_key]
                    })
                    answer = result["answer"]
                    st.markdown(answer)

            st.session_state.chat_histories[history_key].append(
                {"role": "assistant", "content": answer}
            )



    with tab2:
        st.subheader("📊 Workforce Metrics")
        
        # Metrics Row
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Records", len(df))
        c2.metric("Columns", len(df.columns))
        
        num_cols = df.select_dtypes(include=['number']).columns
        if not num_cols.empty:
            c3.metric(f"Avg {num_cols[0].title()}", round(df[num_cols[0]].mean(), 1))
        
        st.divider()
        
        # Visualizations
        obj_cols = df.select_dtypes(include=['object']).columns
        if not obj_cols.empty:
            col_to_plot = obj_cols[0]
            
            # Modern color scheme for charts
            color_scheme = px.colors.sequential.Blues_r
            
            fig = px.pie(
                df, 
                names=col_to_plot, 
                title=f"Distribution by {col_to_plot.title()}",
                color_discrete_sequence=color_scheme,
                hole=0.3  # Donut chart for modern look
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(
                font=dict(size=12),
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig, use_container_width=True)
            
            if not num_cols.empty:
                fig2 = px.bar(
                    df.groupby(col_to_plot)[num_cols[0]].sum().reset_index(), 
                    x=col_to_plot, 
                    y=num_cols[0], 
                    title=f"Total {num_cols[0].title()} by {col_to_plot.title()}",
                    color=num_cols[0],
                    color_continuous_scale='Blues'
                )
                fig2.update_layout(
                    font=dict(size=12),
                    xaxis_title=col_to_plot.title(),
                    yaxis_title=num_cols[0].title()
                )
                st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        st.subheader("📋 Complete Dataset")
        st.dataframe(df, use_container_width=True, height=600)

else:
    # Welcome Screen - Dark Theme Hero Section
    st.markdown("""
        <div style="text-align: center; padding: 60px 20px; margin-top: 40px;">
            <h1 style="color: #4A9EFF; font-size: 3.2rem; font-weight: 800; margin-bottom: 20px; letter-spacing: -1px;">
                💼 Employee Intelligence Assistant
            </h1>
            <p style="color: #9BA3B0; font-size: 1.25rem; margin-bottom: 50px; line-height: 1.6; max-width: 800px; margin-left: auto; margin-right: auto;">
                Transform your workforce data into actionable insights with our high-precision Semantic RAG Engine.
            </p>
            <div style="display: flex; justify-content: center; gap: 24px; flex-wrap: wrap; max-width: 1000px; margin-left: auto; margin-right: auto;">
                <div style="flex: 1; min-width: 240px; padding: 32px; background: #1E2433; border-radius: 20px; border: 1px solid #2D3748; box-shadow: 0 4px 12px rgba(0,0,0,0.3); transition: all 0.3s ease;">
                    <div style="font-size: 3rem; margin-bottom: 16px;">📂</div>
                    <h3 style="color: #E8EAED; margin-bottom: 12px; font-size: 1.25rem; font-weight: 600;">Ingest Data</h3>
                    <p style="color: #9BA3B0; font-size: 0.95rem; line-height: 1.5;">Upload spreadsheets to create isolated intelligence layers.</p>
                </div>
                <div style="flex: 1; min-width: 240px; padding: 32px; background: #1E2433; border-radius: 20px; border: 1px solid #2D3748; box-shadow: 0 4px 12px rgba(0,0,0,0.3); transition: all 0.3s ease;">
                    <div style="font-size: 3rem; margin-bottom: 16px;">🔄</div>
                    <h3 style="color: #E8EAED; margin-bottom: 12px; font-size: 1.25rem; font-weight: 600;">Switch Context</h3>
                    <p style="color: #9BA3B0; font-size: 0.95rem; line-height: 1.5;">Toggle between datasets in your library with zero leakage.</p>
                </div>
                <div style="flex: 1; min-width: 240px; padding: 32px; background: #1E2433; border-radius: 20px; border: 1px solid #2D3748; box-shadow: 0 4px 12px rgba(0,0,0,0.3); transition: all 0.3s ease;">
                    <div style="font-size: 3rem; margin-bottom: 16px;">💬</div>
                    <h3 style="color: #E8EAED; margin-bottom: 12px; font-size: 1.25rem; font-weight: 600;">Ask Anything</h3>
                    <p style="color: #9BA3B0; font-size: 0.95rem; line-height: 1.5;">Query your records using natural language for grounded answers.</p>
                </div>
            </div>
            <div style="margin-top: 60px; padding: 24px; background: linear-gradient(135deg, rgba(74, 158, 255, 0.1) 0%, rgba(74, 158, 255, 0.15) 100%); border-radius: 16px; border: 1px solid rgba(74, 158, 255, 0.3);">
                <p style="color: #4A9EFF; font-size: 1.1rem; font-weight: 500; margin: 0;">
                    ⬅️ Get started by uploading a file in the <strong>Control Panel</strong>
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
