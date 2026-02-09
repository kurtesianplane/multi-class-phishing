import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Phishing Annotation Tool", layout="wide", page_icon="📧")

# Load Google Font
st.markdown('<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">', unsafe_allow_html=True)

# Custom CSS for beautiful compact UI
st.markdown("""
<style>
    html, body, [class*="css"], .stMarkdown, .stButton > button, .stSelectbox, .stTextInput > div > div > input, p, span, div {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    
    .block-container {
        padding: 2.5rem 2rem 0rem 2rem;
        max-width: 100%;
    }
    
    /* Text area styling */
    .stTextArea textarea {
        font-family: 'JetBrains Mono', 'SF Mono', 'Consolas', monospace !important;
        font-size: 13px;
        background: #1a1a2e;
        color: #eee;
        border: 1px solid #333;
        border-radius: 8px;
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        font-size: 13px;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Class buttons special styling */
    .class-btn button {
        font-size: 15px;
        padding: 0.6rem;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    
    section[data-testid="stSidebar"] .stMarkdown {
        color: #e0e0e0;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2);
        border-radius: 10px;
    }
    
    /* Cards/containers */
    .annotation-panel {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid #e9ecef;
    }
    
    /* Reduce vertical spacing */
    div[data-testid="stVerticalBlock"] > div {
        padding-bottom: 0.2rem;
    }
    
    .stSelectbox, .stTextInput {
        font-size: 13px;
    }
    
    /* Status badges */
    .status-done { color: #22c55e; }
    .status-skip { color: #f59e0b; }
    .status-pending { color: #6b7280; }
</style>
""", unsafe_allow_html=True)

CLASS_LABELS = {
    1: "Commercial/R-18",
    2: "Monetary",
    3: "Credential",
    4: "Generic/Deceptive",
}

def init_progress(df, annotator_id):
    df = df.copy()
    # Preserve existing annotations if the uploaded CSV already has them
    if 'annotation_label' not in df.columns:
        df['annotation_label'] = np.nan
    else:
        df['annotation_label'] = pd.to_numeric(df['annotation_label'], errors='coerce')
    if 'annotator_remarks' not in df.columns:
        df['annotator_remarks'] = ''
    else:
        df['annotator_remarks'] = df['annotator_remarks'].fillna('')
    if 'is_skipped' not in df.columns:
        df['is_skipped'] = False
    else:
        df['is_skipped'] = df['is_skipped'].astype(bool)
    df['annotator_id'] = annotator_id
    return df

@st.cache_data
def get_pending_mask(_df):
    return _df['annotation_label'].isna() & ~_df['is_skipped']

def get_next_unannotated(current_idx, total):
    pending = st.session_state.progress_df['annotation_label'].isna() & ~st.session_state.progress_df['is_skipped']
    # Search after current
    after_idx = pending.iloc[current_idx + 1:]
    if after_idx.any():
        return after_idx.idxmax()
    # Wrap to beginning
    before_idx = pending.iloc[:current_idx]
    if before_idx.any():
        return before_idx.idxmax()
    return min(current_idx + 1, total - 1)

def get_prev_index(current_idx):
    return max(0, current_idx - 1)

# Initialize session state
if 'annotator_id' not in st.session_state:
    st.session_state.annotator_id = None
if 'current_idx' not in st.session_state:
    st.session_state.current_idx = 0
if 'progress_df' not in st.session_state:
    st.session_state.progress_df = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

# Sidebar - Annotator Login
st.sidebar.title("Annotator Login")

# File upload section
if not st.session_state.data_loaded:
    st.title("Phishing Email Annotation Tool")
    
    st.markdown("### Upload Dataset")
    st.markdown("CSV should have a text column (`body`, `text`, or `text_cleaned`)")
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        # Auto-detect text column
        text_col = None
        for col in ['body', 'text', 'text_cleaned', 'content', 'email']:
            if col in df.columns:
                text_col = col
                break
        
        if text_col is None:
            st.error("CSV must contain a text column (body, text, text_cleaned, content, or email)")
        else:
            # Standardize column name
            if text_col != 'text_cleaned':
                df['text_cleaned'] = df[text_col]
            
            st.session_state.uploaded_df = df
            st.session_state.data_loaded = True
            st.success(f"Loaded {len(df):,} records (text column: '{text_col}')")
            st.rerun()
    
    st.markdown("### Annotation Classes")
    for k, v in CLASS_LABELS.items():
        st.write(f"**{k}**: {v}")

elif st.session_state.annotator_id is None:
    st.sidebar.markdown("**Select Annotator**")
    ann_cols = st.sidebar.columns(4)
    for i in range(1, 5):
        with ann_cols[i-1]:
            if st.button(f"{i}", key=f"ann_{i}", use_container_width=True):
                st.session_state.annotator_id = f"Annotator_{i}"
                st.session_state.progress_df = init_progress(st.session_state.uploaded_df, st.session_state.annotator_id)
                # Auto-jump to first unannotated item so annotators resume where they left off
                pending = st.session_state.progress_df['annotation_label'].isna() & ~st.session_state.progress_df['is_skipped']
                if pending.any():
                    st.session_state.current_idx = int(pending.idxmax())
                else:
                    st.session_state.current_idx = 0
                st.rerun()
    
    st.title("Phishing Email Annotation Tool")
    st.write("Please select your Annotator ID (1-4) in the sidebar to begin.")
    
    st.markdown("### Annotation Classes")
    for k, v in CLASS_LABELS.items():
        st.write(f"**{k}**: {v}")

else:
    # Sidebar - compact & styled
    st.sidebar.markdown(f"#### 👤 {st.session_state.annotator_id}")
    
    progress_df = st.session_state.progress_df
    current_idx = st.session_state.current_idx
    total = len(progress_df)
    
    # Fast stats
    annotated = int(progress_df['annotation_label'].notna().sum())
    skipped = int(progress_df['is_skipped'].sum())
    remaining = total - annotated - skipped
    pct = (annotated / total) * 100 if total > 0 else 0
    
    st.sidebar.progress(annotated / total if total > 0 else 0)
    st.sidebar.markdown(f"""
    <div style='display:flex;justify-content:space-around;font-size:12px;color:#aaa;margin:-10px 0 15px 0;text-align:center;'>
        <span style='flex:1;'>Done: {annotated:,}</span>
        <span style='flex:1;'>Skip: {skipped:,}</span>
        <span style='flex:1;'>Left: {remaining:,}</span>
        <span style='flex:1;'><b>{pct:.0f}%</b></span>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Classes**")
    st.sidebar.markdown("""
    <div style='font-size:12px;color:#ccc;line-height:1.8;padding-left:5px;'>
    <b>1</b> - Commercial/R-18<br>
    <b>2</b> - Monetary<br>
    <b>3</b> - Credential<br>
    <b>4</b> - Generic/Deceptive
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Actions**")
    c1, c2, c3 = st.sidebar.columns(3)
    with c1:
        if st.button("📋", key="sidebar_skipped", help="View skipped emails"):
            st.session_state.view_skipped = True
    with c2:
        csv_data = progress_df.to_csv(index=False).encode('utf-8')
        st.download_button("💾", csv_data, f"{st.session_state.annotator_id}_annotations.csv", "text/csv", help="Download annotations")
    with c3:
        if st.button("🚪", key="sidebar_logout", help="Logout (download first!)"):
            st.session_state.annotator_id = None
            st.session_state.progress_df = None
            st.session_state.current_idx = 0
            st.rerun()

    row = progress_df.iloc[current_idx]
    
    # Status
    label_val = row['annotation_label']
    is_annotated = pd.notna(label_val)
    if is_annotated:
        status_icon, status_class = "✓", "status-done"
    elif row['is_skipped']:
        status_icon, status_class = ">", "status-skip"
    else:
        status_icon, status_class = "○", "status-pending"
    
    # Header
    source = row.get('source_dataset', 'Unknown')
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);padding:0.8rem 1.2rem;border-radius:10px;color:white;margin-bottom:0.8rem;font-weight:500;display:flex;justify-content:space-between;align-items:center;">
        <span style="font-size:16px;">📧 Email {current_idx + 1} / {total}</span>
        <span style="opacity:0.9;font-size:14px;">{source} · <span class="{status_class}">{status_icon}</span></span>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation row
    nav_cols = st.columns([1, 1, 1, 1, 2, 1])
    with nav_cols[0]:
        if st.button("< Prev", key="btn_prev", use_container_width=True, help="Previous email"):
            st.session_state.current_idx = get_prev_index(current_idx)
            st.rerun()
    with nav_cols[1]:
        if st.button("Next >", key="btn_next", use_container_width=True, help="Next email"):
            st.session_state.current_idx = min(current_idx + 1, total - 1)
            st.rerun()
    with nav_cols[2]:
        if st.button("Skip >", key="btn_skip", use_container_width=True, help="Skip & continue"):
            st.session_state.progress_df.at[current_idx, 'is_skipped'] = True
            st.session_state.current_idx = get_next_unannotated(current_idx, total)
            st.rerun()
    with nav_cols[3]:
        if st.button("Next Blank", key="btn_next_unannotated", use_container_width=True, help="Next unannotated"):
            st.session_state.current_idx = get_next_unannotated(current_idx, total)
            st.rerun()
    with nav_cols[4]:
        jump_to = st.number_input("Go to #", min_value=1, max_value=total, value=current_idx + 1, key="jump_input", label_visibility="collapsed")
    with nav_cols[5]:
        if st.button("Go", key="btn_go", use_container_width=True, help=f"Jump to #{jump_to}"):
            st.session_state.current_idx = jump_to - 1
            st.rerun()
    
    # Main content: Email (left) + Annotation (right)
    left_col, right_col = st.columns([3, 2], gap="medium")
    
    with left_col:
        email_text = str(row['text_cleaned']) if pd.notna(row['text_cleaned']) else ''
        st.text_area("Email Content", value=email_text, height=380, disabled=True, key=f"email_{current_idx}", label_visibility="collapsed")
    
    with right_col:
        # Class selection buttons
        current_label = row['annotation_label']
        selected = int(current_label) if pd.notna(current_label) else None
        
        st.markdown("##### 🏷️ Select Class")
        btn_cols = st.columns(4)
        for i, (k, v) in enumerate(CLASS_LABELS.items()):
            with btn_cols[i]:
                btn_type = "primary" if selected == k else "secondary"
                if st.button(f"{k}", key=f"cls_{k}", use_container_width=True, type=btn_type, help=v):
                    st.session_state.progress_df.at[current_idx, 'annotation_label'] = k
                    st.session_state.progress_df.at[current_idx, 'is_skipped'] = False
                    st.session_state.current_idx = get_next_unannotated(current_idx, total)
                    st.rerun()
        
        # Show selected class name
        if selected:
            st.markdown(f"<div style='text-align:center;color:#667eea;font-size:13px;margin:-5px 0 10px 0;'>Selected: <b>{CLASS_LABELS[selected]}</b></div>", unsafe_allow_html=True)
        
        st.markdown("##### 📝 Notes")
        remarks = st.text_area("", value=row.get('annotator_remarks', '') if pd.notna(row.get('annotator_remarks', '')) else '', height=60, key=f"remarks_{current_idx}", label_visibility="collapsed", placeholder="Optional remarks...")
        
        if remarks and st.button("💾 Save Notes", use_container_width=True, type="secondary"):
            st.session_state.progress_df.at[current_idx, 'annotator_remarks'] = remarks
            st.toast("Notes saved!", icon="✅")
    
    # Skipped emails modal
    if st.session_state.get('view_skipped', False):
        with st.expander("📋 Skipped Emails", expanded=True):
            skipped_df = progress_df[progress_df['is_skipped'] == True]
            if len(skipped_df) > 0:
                st.caption(f"{len(skipped_df)} skipped emails")
                cols = st.columns(8)
                for i, idx in enumerate(skipped_df.index[:24]):
                    with cols[i % 8]:
                        if st.button(f"{idx + 1}", key=f"skip_{idx}", use_container_width=True):
                            st.session_state.current_idx = idx
                            st.session_state.view_skipped = False
                            st.rerun()
            else:
                st.info("No skipped emails")
            if st.button("✕ Close", key="close_skipped"):
                st.session_state.view_skipped = False
                st.rerun()
