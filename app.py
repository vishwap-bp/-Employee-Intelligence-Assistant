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
    page_title="Enterprise Employee AI",
    page_icon="💼",
    layout="wide"
)

# Custom CSS for a professional "Enterprise" look
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stChatItem { border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }
    h1 { color: #2c3e50; }
    .stButton>button { border-radius: 8px; }
    .dataset-card { 
        padding: 10px; 
        border-radius: 10px; 
        border: 1px solid #ddd; 
        margin-bottom: 10px;
        background-color: white;
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
                    
                    # Reset status
                    st.session_state.kb_status = "INACTIVE"
                    if "kb_files" in st.session_state: 
                        del st.session_state.kb_files
                    
                    # Step 2: Force reset of RAG objects
                    if 'rag_chain' in st.session_state:
                        st.session_state.rag_chain = None
                        del st.session_state.rag_chain
                    
                    # Step 3: Clear data references
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
                    
                    # Step 6: Delete all metadata files for this user
                    metadata_parent = user_paths["metadata"]
                    if os.path.exists(metadata_parent):
                        for f in os.listdir(metadata_parent):
                            f_path = os.path.join(metadata_parent, f)
                            if os.path.isfile(f_path):
                                os.remove(f_path)
                    
                    st.session_state.show_factory_confirm = False
                    st.success("✅ Knowledge base cleared!")
                    st.info("💡 **Ready for new data.** You can now upload a fresh file.")
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
        # Per-dataset chat history
        history_key = d["hash"]
        if history_key not in st.session_state.chat_histories:
            st.session_state.chat_histories[history_key] = []
            
        for msg in st.session_state.chat_histories[history_key]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                
        # Create a row with the clear button beside the chat input
        col1, col2 = st.columns([4, 1])
        
        if st.session_state.chat_histories[history_key]:
            with col2:
                if st.button("🗑️", key=f"clear_chat_{history_key}", help="Clear chat history"):
                    st.session_state.chat_histories[history_key] = []
                    st.rerun()
        
        if prompt := st.chat_input(f"Ask about {d['filename']}...", key="chat_input"):
            st.session_state.chat_histories[history_key].append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
                
            with st.chat_message("assistant"):
                with st.spinner("Analyzing..."):
                    # Dynamic RAG connection
                    chain = get_rag_chain(api_key, db_path=d["db_path"], username=st.session_state.username)
                    result = chain.invoke({"input": prompt})
                    answer = result["answer"]
                    st.markdown(answer)
                    st.session_state.chat_histories[history_key].append({"role": "assistant", "content": answer})

    with tab2:
        st.subheader("Workforce Metrics")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Records", len(df))
        c2.metric("Columns", len(df.columns))
        
        num_cols = df.select_dtypes(include=['number']).columns
        if not num_cols.empty:
            c3.metric(f"Avg {num_cols[0].title()}", round(df[num_cols[0]].mean(), 1))
        
        st.divider()
        
        obj_cols = df.select_dtypes(include=['object']).columns
        if not obj_cols.empty:
            col_to_plot = obj_cols[0]
            fig = px.pie(df, names=col_to_plot, title=f"Distribution by {col_to_plot.title()}")
            st.plotly_chart(fig, use_container_width=True)
            
            if not num_cols.empty:
                fig2 = px.bar(df.groupby(col_to_plot)[num_cols[0]].sum().reset_index(), 
                             x=col_to_plot, y=num_cols[0], title=f"Total {num_cols[0].title()} by {col_to_plot.title()}")
                st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        st.dataframe(df, use_container_width=True)

else:
    # Welcome Screen - Final Polished Hero Section
    st.markdown("""
        <div style="text-align: center; padding: 50px 20px; margin-top: 20px;">
            <h1 style="color: #1c83e1; font-size: 3rem; font-weight: 800; margin-bottom: 15px;">💼 Employee Intelligence Assistant</h1>
            <p style="color: #f5f5dc; font-size: 1.15rem; margin-bottom: 45px; width: 100%; white-space: nowrap; line-height: 1.6;">
                Transform your workforce data into actionable insights with our high-precision Semantic RAG Engine.
            </p>
            <div style="display: flex; justify-content: center; gap: 20px; flex-wrap: wrap; max-width: 1000px; margin-left: auto; margin-right: auto;">
                <div style="flex: 1; min-width: 220px; padding: 25px; background-color: rgba(28, 131, 225, 0.04); border-radius: 18px; border: 1px solid rgba(28, 131, 225, 0.1);">
                    <div style="font-size: 2.5rem; margin-bottom: 12px; color: #5DA9FF;">📂</div>
                    <h3 style="color: #5DA9FF; margin-bottom: 8px; font-size: 1.2rem;">Ingest Data</h3>
                    <p style="color: #B0B7C3; font-size: 0.9rem; line-height: 1.4; font-weight: 500;">Upload spreadsheets to create isolated intelligence layers.</p>
                </div>
                <div style="flex: 1; min-width: 220px; padding: 25px; background-color: rgba(40, 167, 69, 0.04); border-radius: 18px; border: 1px solid rgba(40, 167, 69, 0.1);">
                    <div style="font-size: 2.5rem; margin-bottom: 12px; color: #5DA9FF;">🔄</div>
                    <h3 style="color: #5DA9FF; margin-bottom: 8px; font-size: 1.2rem;">Switch Context</h3>
                    <p style="color: #B0B7C3; font-size: 0.9rem; line-height: 1.4; font-weight: 500;">Toggle between datasets in your library with zero leakage.</p>
                </div>
                <div style="flex: 1; min-width: 220px; padding: 25px; background-color: rgba(23, 162, 184, 0.04); border-radius: 18px; border: 1px solid rgba(23, 162, 184, 0.1);">
                    <div style="font-size: 2.5rem; margin-bottom: 12px; color: #5DA9FF;">💬</div>
                    <h3 style="color: #5DA9FF; margin-bottom: 8px; font-size: 1.2rem;">Ask Anything</h3>
                    <p style="color: #B0B7C3; font-size: 0.9rem; line-height: 1.4; font-weight: 500;">Query your records using natural language for grounded answers.</p>
                </div>
            </div>
            <div style="margin-top: 50px;">
                <p style="color: #90a4ae; font-size: 1rem;">⬅️ Get started by uploading a file in the <b>Control Center</b></p>
            </div>
        </div>
    """, unsafe_allow_html=True)
