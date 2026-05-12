"""
RAG Vector Search Assessment - Streamlit Web App

A beautiful, interactive web interface to visualize the RAG assessment results.
"""

import streamlit as st
import json
import os
from pathlib import Path
import pandas as pd

# ============================================
# PAGE CONFIG & STYLING
# ============================================

st.set_page_config(
    page_title="RAG Assessment Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        color: #ffffff;
    }
    
    /* Cards */
    .stMetric {
        background: rgba(255,255,255,0.1);
        border-radius: 15px;
        padding: 20px;
        backdrop-filter: blur(10px);
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #00d4ff !important;
        font-weight: 700 !important;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: rgba(0,0,0,0.3);
        backdrop-filter: blur(20px);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #00d4ff, #7b2cbf);
        border: none;
        border-radius: 25px;
        padding: 10px 30px;
        font-weight: bold;
    }
    
    /* Dataframes */
    .stDataFrame {
        background: rgba(255,255,255,0.05);
        border-radius: 15px;
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background: rgba(255,255,255,0.1);
        border-radius: 10px;
        color: #00d4ff;
    }
    
    /* Input fields */
    .stTextInput > div > div {
        background: rgba(255,255,255,0.1);
        border-radius: 10px;
        border: 1px solid #00d4ff;
    }
    
    /* Info boxes */
    .stInfo {
        background: rgba(0,212,255,0.2);
        border-radius: 10px;
        border-left: 4px solid #00d4ff;
    }
    
    /* Success boxes */
    .stSuccess {
        background: rgba(0,255,127,0.2);
        border-radius: 10px;
        border-left: 4px solid #00ff7f;
    }
    
    /* Warning boxes */
    .stWarning {
        background: rgba(255,165,0,0.2);
        border-radius: 10px;
        border-left: 4px solid #ffa500;
    }
    
    /* Tables */
    div[data-testid="stTable"] {
        background: rgba(255,255,255,0.05);
        border-radius: 15px;
        padding: 20px;
    }
    
    /* Metrics */
    div[data-testid="stMetricValue"] {
        color: #00ff7f !important;
        font-size: 2rem !important;
    }
    
    div[data-testid="stMetricLabel"] {
        color: #ffffff !important;
    }
    
    /* Divider */
    hr {
        border-color: rgba(0,212,255,0.3);
    }
    
    /* Custom card styling */
    .custom-card {
        background: linear-gradient(135deg, rgba(0,212,255,0.2), rgba(123,44,191,0.2));
        border-radius: 20px;
        padding: 25px;
        margin: 10px 0;
        border: 1px solid rgba(0,212,255,0.3);
        backdrop-filter: blur(10px);
    }
    
    .metric-box {
        background: linear-gradient(135deg, rgba(0,255,127,0.15), rgba(0,212,255,0.15));
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        border: 1px solid rgba(0,212,255,0.3);
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# PATH SETUP
# ============================================

DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")
DOCS_FILE = DATA_DIR / "documents.json"
EVAL_FILE = DATA_DIR / "eval_pairs.json"
RESULTS_FILE = OUTPUT_DIR / "results.json"

# ============================================
# DATA LOADING FUNCTIONS
# ============================================

@st.cache_data
def load_results():
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE, "r") as f:
            return json.load(f)
    return None

@st.cache_data
def load_documents():
    if DOCS_FILE.exists():
        with open(DOCS_FILE, "r") as f:
            return json.load(f)
    return []

@st.cache_data
def load_eval_pairs():
    if EVAL_FILE.exists():
        with open(EVAL_FILE, "r") as f:
            return json.load(f)
    return []

# ============================================
# MAIN APP
# ============================================

def main():
    # Header
    st.markdown("""
    <div style="text-align: center; padding: 20px;">
        <h1 style="font-size: 3rem; margin-bottom: 10px;">🔍 RAG Vector Search Assessment</h1>
        <p style="font-size: 1.3rem; color: #7b2cbf;">Local RAG System with FAISS + Semantic Embeddings</p>
        <p style="color: #00d4ff;">Powered by all-MiniLM-L6-v2 + Streamlit</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, rgba(0,212,255,0.2), rgba(123,44,191,0.2)); border-radius: 20px;">
            <h2 style="color: #00d4ff;">📊 Navigation</h2>
        </div>
        """, unsafe_allow_html=True)
        
        page = st.radio(
            "",
            ["🏠 Home", "📈 Results", "🔎 Search", "📋 Data", "ℹ️ About"]
        )
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Quick stats in sidebar
        st.markdown("""
        <div style="background: rgba(255,255,255,0.1); border-radius: 15px; padding: 15px;">
            <h4 style="color: #00d4ff;">📊 Quick Stats</h4>
        </div>
        """, unsafe_allow_html=True)
        
        docs = load_documents()
        evals = load_eval_pairs()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📄 Docs", len(docs))
        with col2:
            st.metric("❓ Queries", len(evals))
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # GitHub link
        st.markdown("""
        <div style="text-align: center; padding: 15px;">
            <a href="https://github.com/badalsharma9929/rag-assessment" target="_blank" 
               style="background: linear-gradient(90deg, #00d4ff, #7b2cbf); 
                      color: white; padding: 12px 25px; border-radius: 25px; 
                      text-decoration: none; font-weight: bold;">
               🔗 GitHub Repository
            </a>
        </div>
        """, unsafe_allow_html=True)
    
    # Route to pages
    if page == "🏠 Home":
        show_home()
    elif page == "📈 Results":
        show_results()
    elif page == "🔎 Search":
        show_search()
    elif page == "📋 Data":
        show_data()
    else:
        show_about()


def show_home():
    """Home page with overview."""
    st.markdown("""
    <div class="custom-card" style="text-align: center;">
        <h2 style="color: #00ff7f;">✅ System Status: Operational</h2>
        <p style="font-size: 1.2rem;">This RAG assessment demonstrates semantic search using vector embeddings.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Features
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-box">
            <h3 style="font-size: 2.5rem;">🔍</h3>
            <h4>Semantic Search</h4>
            <p>Real embeddings</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-box">
            <h3 style="font-size: 2.5rem;">💾</h3>
            <h4>FAISS Storage</h4>
            <p>Fast similarity</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-box">
            <h3 style="font-size: 2.5rem;">📊</h3>
            <h4>Benchmark</h4>
            <p>P@K, R@K, MRR</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-box">
            <h3 style="font-size: 2.5rem;">☁️</h3>
            <h4>GCP Mock</h4>
            <p>No API needed</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Quick overview
    st.subheader("📋 Latest Results Summary")
    
    results = load_results()
    if results:
        metrics = results.get("metrics", {})
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Precision@5", f"{metrics.get('precision_5', {}).get('local', 0):.2%}")
        with col2:
            st.metric("Recall@5", f"{metrics.get('recall_5', {}).get('local', 0):.2%}")
        with col3:
            st.metric("MRR", f"{metrics.get('mrr', {}).get('local', 0):.2%}")
        with col4:
            st.metric("NDCG@5", f"{metrics.get('ndcg_5', {}).get('local', 0):.2%}")
        
        st.info("💡 Navigate to '📈 Results' to see the full comparison with GCP mock!")
    else:
        st.warning("⚠️ No results found. Please run `python main.py` first.")


def show_results():
    """Results page with detailed metrics."""
    st.header("📈 Benchmark Results")
    
    results = load_results()
    
    if not results:
        st.error("❌ No results found! Please run `python main.py` first.")
        return
    
    # Model info cards
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="custom-card" style="border-left: 4px solid #00ff7f;">
            <h3 style="color: #00ff7f;">🟢 Local Model</h3>
            <p style="font-size: 1.1rem;">{results['local_model']}</p>
            <p style="color: #aaa;">Uses real semantic embeddings</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="custom-card" style="border-left: 4px solid #ffa500;">
            <h3 style="color: #ffa500;">🟠 GCP Mock</h3>
            <p style="font-size: 1.1rem;">{results['gcp_model']}</p>
            <p style="color: #aaa;">Simulated with random embeddings</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Key metrics comparison
    st.subheader("🎯 Key Performance Metrics")
    
    metrics = results.get("metrics", {})
    
    # Create comparison table
    table_data = []
    key_metrics = ["precision_5", "recall_5", "mrr", "ndcg_5", "precision_1", "precision_3"]
    
    for metric in key_metrics:
        if metric in metrics:
            local = metrics[metric]["local"]
            gcp = metrics[metric]["gcp"]
            diff = local - gcp
            
            # Format name
            name = metric.replace("_", " ").title()
            if "mrr" in metric:
                name = "MRR"
            elif "precision" in metric:
                name = f"P@{metric.split('_')[1]}"
            elif "recall" in metric:
                name = f"R@{metric.split('_')[1]}"
            elif "ndcg" in metric:
                name = f"NDCG@{metric.split('_')[1]}"
            
            table_data.append({
                "Metric": f"**{name}**",
                "Local (✅)": f"```{local:.4f}```",
                "GCP Mock (⚠️)": f"```{gcp:.4f}```",
                "Difference": f"**{diff:+.4f}**"
            })
    
    if table_data:
        df = pd.DataFrame(table_data)
        st.markdown(df.to_markdown(index=False), unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Visual chart
    st.subheader("📊 Visual Comparison")
    
    local_scores = [
        metrics.get("precision_5", {}).get("local", 0),
        metrics.get("recall_5", {}).get("local", 0),
        metrics.get("mrr", {}).get("local", 0),
        metrics.get("ndcg_5", {}).get("local", 0)
    ]
    
    gcp_scores = [
        metrics.get("precision_5", {}).get("gcp", 0),
        metrics.get("recall_5", {}).get("gcp", 0),
        metrics.get("mrr", {}).get("gcp", 0),
        metrics.get("ndcg_5", {}).get("gcp", 0)
    ]
    
    chart_df = pd.DataFrame({
        "Metric": ["Precision@5", "Recall@5", "MRR", "NDCG@5"],
        "Local Model": local_scores,
        "GCP Mock": gcp_scores
    })
    
    st.bar_chart(chart_df.set_index("Metric"))
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # All metrics expander
    with st.expander("📋 View All Metrics (P@K, R@K, NDCG@K)"):
        all_metrics = []
        for metric, values in sorted(metrics.items()):
            all_metrics.append({
                "Metric": metric,
                "Local": f"{values['local']:.4f}",
                "GCP": f"{values['gcp']:.4f}",
                "Diff": f"{values['local']-values['gcp']:+.4f}"
            })
        
        if all_metrics:
            df_all = pd.DataFrame(all_metrics)
            st.table(df_all)
    
    st.success("""
    💡 The local model significantly outperforms the GCP mock because it uses 
    real semantic embeddings that understand meaning, while the mock uses 
    random vectors based on text hash.
    """)


def show_search():
    """Document search page."""
    st.header("🔎 Document Explorer")
    
    documents = load_documents()
    
    if not documents:
        st.error("❌ No documents found!")
        return
    
    st.success(f"📄 Loaded {len(documents)} documents")
    
    # Search/filter
    search = st.text_input("🔍 Search documents by title or concept:", placeholder="e.g., neural networks, machine learning...")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Filter documents
    if search:
        filtered = [d for d in documents if search.lower() in d.get('title', '').lower() or 
                   search.lower() in d.get('concept', '').lower()]
    else:
        filtered = documents
    
    st.markdown(f"**Showing {len(filtered)} documents**")
    
    # Display documents in cards
    for doc in filtered:
        with st.expander(f"📄 {doc.get('title', 'Untitled')}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**ID:** `{doc.get('id', 'N/A')}`")
                st.markdown(f"**Topic:** {doc.get('topic', 'N/A')}")
            
            with col2:
                st.markdown(f"**Concept:** {doc.get('concept', 'N/A')}")
                st.markdown(f"**Words:** {doc.get('word_count', 'N/A')}")
            
            st.markdown("---")
            st.markdown(doc.get('text', '')[:600] + "...")


def show_data():
    """Evaluation data page."""
    st.header("📋 Evaluation Data")
    
    eval_pairs = load_eval_pairs()
    
    if not eval_pairs:
        st.error("❌ No evaluation data found!")
        return
    
    st.success(f"❓ Loaded {len(eval_pairs)} evaluation query-document pairs")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Stats
    col1, col2, col3 = st.columns(3)
    
    total_relevant = sum(pair.get('n_relevant', 0) for pair in eval_pairs)
    
    with col1:
        st.metric("Total Pairs", len(eval_pairs))
    with col2:
        st.metric("Total Relevant Docs", total_relevant)
    with col3:
        st.metric("Avg Relevant/Doc", f"{total_relevant/len(eval_pairs):.1f}")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Show pairs
    st.subheader("Query-Document Pairs")
    
    for pair in eval_pairs:
        with st.expander(f"❓ {pair.get('query', 'N/A')[:70]}..."):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**ID:** `{pair.get('id', 'N/A')}`")
                st.markdown(f"**Query:** {pair.get('query', 'N/A')}")
            
            with col2:
                st.markdown(f"**Relevant Docs:** {', '.join(pair.get('relevant_docs', []))}")
                st.markdown(f"**Concept:** {pair.get('concept', 'N/A')}")


def show_about():
    """About page."""
    st.header("ℹ️ About This Project")
    
    st.markdown("""
    <div class="custom-card">
        <h2 style="color: #00d4ff;">🔍 RAG Vector Search Assessment</h2>
        <p style="font-size: 1.1rem;">
            A local Retrieval-Augmented Generation (RAG) system that demonstrates 
            semantic search using vector embeddings with FAISS.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Features
    st.subheader("✨ Features")
    
    features = [
        ("🔍", "Semantic Search", "Using all-MiniLM-L6-v2 embeddings"),
        ("💾", "FAISS Vector Store", "Fast similarity search"),
        ("☁️", "GCP Mock", "Simulates Vertex AI without API keys"),
        ("📊", "Benchmark Metrics", "P@K, R@K, MRR, NDCG@K"),
        ("🌐", "Web UI", "Beautiful Streamlit interface"),
        ("✅", "Tests", "Comprehensive pytest test suite")
    ]
    
    for icon, title, desc in features:
        st.markdown(f"""
        <div style="display: flex; align-items: center; padding: 10px; 
                    background: rgba(255,255,255,0.05); border-radius: 10px; margin: 5px 0;">
            <span style="font-size: 1.5rem; margin-right: 15px;">{icon}</span>
            <div>
                <strong style="color: #00d4ff;">{title}</strong><br>
                <span style="color: #aaa;">{desc}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Tech stack
    st.subheader("🛠️ Technology Stack")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        - **Python 3.9+**
        - **Transformers** (all-MiniLM-L6-v2)
        - **FAISS** (vector search)
        - **NumPy** (numerical computing)
        """)
    
    with col2:
        st.markdown("""
        - **Streamlit** (web UI)
        - **Pandas** (data handling)
        - **Pytest** (testing)
        - **Tabulate** (output formatting)
        """)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # How to run
    st.subheader("🚀 How to Run")
    
    st.code("""
# Clone the repository
git clone https://github.com/badalsharma9929/rag-assessment.git
cd rag-assessment

# Install dependencies
pip install -r requirements.txt

# Run the assessment
python main.py

# Run the web app
streamlit run streamlit_app.py
    """, language="bash")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # GitHub link
    st.success("""
    📂 **Repository:** https://github.com/badalsharma9929/rag-assessment
    
    ⭐ If you find this helpful, please star the repo!
    """)


# ============================================
# RUN APP
# ============================================

if __name__ == "__main__":
    main()