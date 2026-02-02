import fix_sqlite # Must be first
import os
import streamlit as st
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv
from streamlit_extras.metric_cards import style_metric_cards

# Importing your logic modules
from processor import clean_and_serialize
from rag_engine import get_rag_chain
from config import PERSIST_DIRECTORY, HASH_FILE, get_active_db_path

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
    </style>
    """, unsafe_allow_html=True)

st.title("💼 Employee Intelligence Assistant")
st.caption("Professional RAG-based analysis for HR and Project Management")

# 2.5 Robust System Initialization (Startup Only)
if "kb_status" not in st.session_state:
    st.session_state.kb_status = "INACTIVE"
    active_db_path = os.path.abspath(get_active_db_path())
    kb_db_file = os.path.join(active_db_path, "chroma.sqlite3")
    
    if os.path.exists(kb_db_file):
        # Physical DB exists, now attempt to load visuals/intelligence
        try:
            # 1. Connect RAG
            if api_key:
                st.session_state.rag_chain = get_rag_chain(api_key)
            
            # 2. Load Visuals (CSV)
            df_path = os.path.join(os.path.dirname(HASH_FILE), "active_data.csv")
            if os.path.exists(df_path):
                st.session_state.df = pd.read_csv(df_path)
            
            # 3. Detect Metadata State
            if os.path.exists(HASH_FILE):
                with open(HASH_FILE, 'r') as f:
                    import json
                    data = json.load(f)
                    filenames = data.get('filenames', [])
                    st.session_state.kb_files = filenames
                    
                    if "df" in st.session_state:
                        st.session_state.kb_status = "ACTIVE"
                    else:
                        st.session_state.kb_status = "MIGRATION_REQUIRED"
            else:
                st.session_state.kb_status = "ACTIVE_LEGACY"
                
        except Exception as e:
            # If initialization fails, fallback to migration required
            st.session_state.kb_status = "MIGRATION_REQUIRED"

# Helper for main area logic
kb_exists = st.session_state.kb_status != "INACTIVE"

# 3. Sidebar Controls
with st.sidebar:
    st.markdown("### Data Control Center")
    
    uploaded_file = st.file_uploader(
        "Upload Team Dataset (CSV or Excel)",
        type=["csv", "xlsx"],
        help="Upload the employee spreadsheet you want to analyze."
    )
    
    st.divider()
    
    # Knowledge Base Management
    st.markdown("### 🗄️ Knowledge Base")
    
    kb_status = st.session_state.get("kb_status", "INACTIVE")
    
    if kb_status == "ACTIVE":
        st.success("✅ Intelligence Layer: Active")
        files = st.session_state.get("kb_files", [])
        st.caption(f"Tracking {len(files)} dataset{'s' if len(files) > 1 else ''}")
        with st.expander("📄 View Active Files"):
            for name in files:
                st.text(f"• {name}")
    elif kb_status in ["ACTIVE_LEGACY", "MIGRATION_REQUIRED"]:
        st.warning("⚠️ Sync Required")
        st.info("The brain is ready, but your dashboard visuals need a one-time sync. **Please upload your file.**")
    else:
        st.info("📭 No knowledge base yet")
    
    # Clear KB button (if any data exists)
    if kb_status != "INACTIVE":
        if st.button("🗑️ Clear Knowledge Base", type="secondary", use_container_width=True):
            st.session_state.show_clear_confirm = True
        
        # Confirmation dialog
        if st.session_state.get('show_clear_confirm', False):
            st.warning("⚠️ This will delete all ingested data!")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("✅ Confirm", type="primary", use_container_width=True):
                    try:
                        import shutil
                        import time
                        import gc
                        
                        # Step 1: Clear Streamlit Resource Cache (Critical for releasing handles)
                        st.cache_resource.clear()
                        
                        # Reset status
                        st.session_state.kb_status = "INACTIVE"
                        if "kb_files" in st.session_state: del st.session_state.kb_files
                        
                        # Step 2: Force reset of RAG objects
                        if 'rag_chain' in st.session_state:
                            st.session_state.rag_chain = None
                            del st.session_state.rag_chain
                        
                        # Step 3: Clear data references
                        if 'df' in st.session_state:
                            st.session_state.df = None
                            del st.session_state.df
                        
                        # Set a flag to prevent immediate re-ingestion of the same file
                        st.session_state.ignore_current_file = st.session_state.get('active_file')
                        
                        # Step 4: Clear active file
                        if 'active_file' in st.session_state:
                            del st.session_state.active_file
                        
                        # Step 5: Ultra-aggressive garbage collection
                        for _ in range(5):
                            gc.collect()
                        time.sleep(2.0)  # Wait longer for OS to catch up
                        
                        # Step 6: Try to delete with retry logic
                        max_retries = 3
                        # Use the actual active path (Function now available globally)
                        target_dir = os.path.abspath(get_active_db_path())
                        
                        for attempt in range(max_retries):
                            try:
                                # Delete vector database
                                if os.path.exists(target_dir):
                                    shutil.rmtree(target_dir)
                                break  # Success!
                            except (PermissionError, OSError) as e:
                                if attempt < max_retries - 1:
                                    time.sleep(2.0) 
                                    gc.collect()
                                else:
                                    # Fallback: rename directory if it can't be deleted
                                    if os.path.exists(PERSIST_DIRECTORY):
                                        try:
                                            os.rename(PERSIST_DIRECTORY, f"{PERSIST_DIRECTORY}_old_{int(time.time())}")
                                        except:
                                            raise e
                        
                        # Step 7: Delete hash and csv files
                        if os.path.exists(HASH_FILE):
                            os.remove(HASH_FILE)
                        df_path = os.path.join(os.path.dirname(HASH_FILE), "active_data.csv")
                        if os.path.exists(df_path):
                            os.remove(df_path)
                        
                        st.session_state.show_clear_confirm = False
                        st.success("✅ Knowledge base cleared!")
                        st.info("💡 **Ready for new data.** You can now upload a fresh file.")
                        st.rerun() 
                        
                    except Exception as e:
                        st.error(f"⚠️ Could not fully clear KB: {e}")
                        st.warning("**Workaround**: Restart the application or refresh the browser.")
            
            with col2:
                if st.button("❌ Cancel", use_container_width=True):
                    st.session_state.show_clear_confirm = False
                    st.rerun()
    
    st.divider()
    
    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# 4. API Check & Logic
if not api_key:
    st.error("🔑 Groq API Key missing. Please check your .env file or create one with GROQ_API_KEY=your_key")
    st.stop()

if uploaded_file:
    # Logic to only process when a NEW file is uploaded
    is_new_file = "active_file" not in st.session_state or st.session_state.active_file != uploaded_file.name
    was_just_cleared = st.session_state.get('ignore_current_file') == uploaded_file.name
    
    if is_new_file and not was_just_cleared:
        try:
            with st.status("🛠️ Building Semantic Intelligence Layer...", expanded=True) as status:
                # 1. Read Bytes & Reset Pointer (Critical for Hashing + Pandas)
                bytes_data = uploaded_file.read()
                uploaded_file.seek(0)
                
                # 2. Ingest (Check Hash -> Skip if exists -> Embed if new)
                from ingest import ingest_dataset
                st.write("Verifying dataset signature...")
                ingest_status = ingest_dataset(uploaded_file, bytes_data)
                
                if ingest_status == "EXISTING":
                    st.info("✅ Dataset recognized! Loading existing intelligence from disk...")
                    # IMPORTANT: Force metadata update so parquet is created for legacy data
                    from ingest import get_file_hash, save_dataset_hash
                    from config import get_active_db_path, HASH_FILE
                    
                    # Get the DF first
                    uploaded_file.seek(0)
                    _, _, df = clean_and_serialize(uploaded_file)
                    st.session_state.df = df
                    
                    current_hash = get_file_hash(bytes_data)
                    active_path = get_active_db_path()
                    save_dataset_hash(current_hash, active_path, uploaded_file.name, df)
                else:
                    st.write("Generating vector embeddings...")
                
                # 3. Load RAG Chain (Connects to the persistent DB)
                st.session_state.rag_chain = get_rag_chain(api_key)
                
                # 4. Preview Data (Is already done for EXISTING above, but we need it for NEW)
                if ingest_status != "EXISTING":
                    uploaded_file.seek(0)
                    _, _, df = clean_and_serialize(uploaded_file)
                    st.session_state.df = df
                
                st.session_state.active_file = uploaded_file.name
                
                # Update global status so sidebar/main area refresh
                st.session_state.kb_status = "ACTIVE"
                from ingest import HASH_FILE
                if os.path.exists(HASH_FILE):
                    try:
                        with open(HASH_FILE, 'r') as f:
                            data = json.load(f)
                            st.session_state.kb_files = data.get('filenames', [])
                    except: pass

                status.update(label="✅ System Ready", state="complete", expanded=False)
                st.rerun() # Refresh to update sidebar KB status
        except Exception as e:
            st.error(f"Failed to process file: {e}")
            st.stop()

# 5. Main Interface TABS
if "df" in st.session_state:
    tab1, tab2, tab3 = st.tabs(["💬 AI Chat", "📈 Executive Dashboard", "📊 Raw Data"])
    
    # TAB 1: Chat Interface
    with tab1:
        st.subheader("Natural Language Analysis")
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if query := st.chat_input("Ask about workloads, top performers, or specific employees..."):
            st.session_state.messages.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)

            with st.chat_message("assistant"):
                with st.spinner("Analyzing records..."):
                    try:
                        result = st.session_state.rag_chain.invoke({"input": query})
                        answer = result["answer"]
                        st.markdown(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    except Exception as e:
                        st.error(f"The AI encountered an error: {e}")

    # TAB 2: Dashboard (Plotly)
    with tab2:
        st.subheader("Workforce Analytics")
        df = st.session_state.df
        
        # Metric Cards with error handling
        try:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="📊 Total Records",
                    value=f"{len(df):,}"
                )
            
            with col2:
                st.metric(
                    label="📋 Columns Analyzed",
                    value=len(df.columns)
                )
            
            with col3:
                # Try to find a numeric column for average
                numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
                if len(numeric_cols) > 0:
                    first_num_col = numeric_cols[0]
                    avg_val = df[first_num_col].mean()
                    st.metric(
                        label=f"📈 Avg {first_num_col.replace('_', ' ').title()}",
                        value=f"{avg_val:.1f}"
                    )
                else:
                    # Fallback: show unique values count
                    obj_cols = df.select_dtypes(include=['object']).columns
                    if len(obj_cols) > 0:
                        unique_count = df[obj_cols[0]].nunique()
                        st.metric(
                            label=f"🔢 Unique {obj_cols[0].replace('_', ' ').title()}",
                            value=unique_count
                        )
            
            # Apply custom styling for visibility and premium look
            st.markdown("""
                <style>
                [data-testid="stMetric"] {
                    background-color: rgba(28, 131, 225, 0.15);
                    padding: 20px;
                    border-radius: 12px;
                    border: 1px solid rgba(28, 131, 225, 0.3);
                    box-shadow: 2px 2px 10px rgba(0,0,0,0.2);
                    text-align: center;
                }
                [data-testid="stMetricLabel"] {
                    color: #1c83e1 !important;
                    font-weight: 700 !important;
                    font-size: 1.1rem !important;
                }
                [data-testid="stMetricValue"] {
                    color: #ffffff !important;
                    font-size: 2rem !important;
                    font-weight: 800 !important;
                }
                </style>
            """, unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"Error displaying metrics: {e}")

        st.divider()

        # Dynamic Charts
        try:
            obj_cols = df.select_dtypes(include=['object']).columns
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            
            if len(obj_cols) > 0:
                # Chart 1: Pie chart of first categorical column
                first_cat_col = obj_cols[0]
                value_counts = df[first_cat_col].value_counts().head(10)
                
                fig1 = px.pie(
                    values=value_counts.values,
                    names=value_counts.index,
                    title=f"Distribution by {first_cat_col.replace('_', ' ').title()}"
                )
                st.plotly_chart(fig1, use_container_width=True)

            if len(obj_cols) > 0 and len(numeric_cols) > 0:
                # Chart 2: Bar chart
                first_num_col = numeric_cols[0]
                first_cat_col = obj_cols[0]
                
                grouped = df.groupby(first_cat_col)[first_num_col].sum().sort_values(ascending=False).head(10)
                
                fig2 = px.bar(
                    x=grouped.index,
                    y=grouped.values,
                    title=f"{first_num_col.replace('_', ' ').title()} by {first_cat_col.replace('_', ' ').title()}",
                    labels={'x': first_cat_col.replace('_', ' ').title(), 'y': first_num_col.replace('_', ' ').title()}
                )
                st.plotly_chart(fig2, use_container_width=True)
                
        except Exception as e:
            st.warning(f"Could not generate charts: {e}")


    # TAB 3: Data Grid
    with tab3:
        st.dataframe(st.session_state.df, use_container_width=True)

elif st.session_state.kb_status in ["ACTIVE", "ACTIVE_LEGACY", "MIGRATION_REQUIRED"]:
    # DB exists but no DF (Session refresh or initial legacy state)
    st.warning("🔄 **Information Sync in Progress**")
    st.info("""
        The Knowledge Base is active, but your dashboard visuals haven't been synchronized for this session yet.
        
        **To fix this:**
        1. Simply **select your dataset file** in the sidebar.
        2. The system will instantly link your visuals to the active Intelligence Layer.
    """)
    
    # Guide points
    st.markdown("""
        ---
        ### 🛠️ Why am I seeing this?
        You previously built an Intelligence Layer, but to see the Dashboard charts after a refresh, 
        the app needs to see the file once more to 're-draw' the visuals.
    """)
else:
    # No data at all
    st.info("👋 **Welcome! Follow these steps to analyze your team data:**")
    st.markdown("""
    1. 📂 **Upload File**: Select an Employee CSV or Excel file from the sidebar.
    2. ⚙️ **Wait for Ingestion**: The AI will clean and "learn" your dataset semantically.
    3. 💬 **Ask Questions**: Use the chat box to ask about your team.
    """)
