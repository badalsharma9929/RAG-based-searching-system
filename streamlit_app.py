"""
Streamlit Web App for RAG Vector Search Assessment

A simple web interface to visualize and interact with the RAG assessment results.
"""

import streamlit as st
import json
import os
from pathlib import Path

# Page config
st.set_page_config(
    page_title="RAG Assessment Dashboard",
    page_icon="🔍",
    layout="wide"
)

# Paths
DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")
DOCS_FILE = DATA_DIR / "documents.json"
EVAL_FILE = DATA_DIR / "eval_pairs.json"
RESULTS_FILE = OUTPUT_DIR / "results.json"


def load_results():
    """Load benchmark results from JSON."""
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE, "r") as f:
            return json.load(f)
    return None


def load_documents():
    """Load documents for search."""
    if DOCS_FILE.exists():
        with open(DOCS_FILE, "r") as f:
            return json.load(f)
    return []


def load_eval_pairs():
    """Load evaluation pairs."""
    if EVAL_FILE.exists():
        with open(EVAL_FILE, "r") as f:
            return json.load(f)
    return []


def main():
    """Main Streamlit app."""
    
    # Header
    st.title("🔍 RAG Vector Search Assessment")
    st.markdown("### Local RAG System with FAISS + Embeddings")
    
    # Sidebar
    st.sidebar.header("📊 Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["📈 Results Dashboard", "🔎 Document Search", "📋 Evaluation Data", "ℹ️ About"]
    )
    
    if page == "📈 Results Dashboard":
        show_results_dashboard()
    elif page == "🔎 Document Search":
        show_document_search()
    elif page == "📋 Evaluation Data":
        show_eval_data()
    else:
        show_about()


def show_results_dashboard():
    """Show benchmark results dashboard."""
    st.header("📈 Benchmark Results")
    
    results = load_results()
    
    if not results:
        st.error("No results found! Please run `python main.py` first.")
        return
    
    # Model info
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**Local Model:** {results['local_model']}")
    with col2:
        st.warning(f"**GCP Model:** {results['gcp_model']}")
    
    st.markdown("---")
    
    # Metrics comparison
    st.subheader("📊 Performance Metrics Comparison")
    
    metrics = results.get("metrics", {})
    
    # Create a nice table
    table_data = []
    for metric in ["precision_5", "recall_5", "mrr", "ndcg_5"]:
        if metric in metrics:
            local_val = metrics[metric]["local"]
            gcp_val = metrics[metric]["gcp"]
            diff = local_val - gcp_val
            
            # Format metric name
            name = metric.replace("_", " ").title()
            if "mrr" in metric:
                name = "MRR (Mean Reciprocal Rank)"
            elif "ndcg" in metric:
                name = f"NDCG@{metric.split('_')[1]}"
            elif "precision" in metric:
                name = f"Precision@{metric.split('_')[1]}"
            elif "recall" in metric:
                name = f"Recall@{metric.split('_')[1]}"
            
            table_data.append({
                "Metric": name,
                "Local": f"{local_val:.4f}",
                "GCP Mock": f"{gcp_val:.4f}",
                "Difference": f"{diff:+.4f}"
            })
    
    # Display as dataframe
    if table_data:
        import pandas as pd
        df = pd.DataFrame(table_data)
        st.table(df)
    
    st.markdown("---")
    
    # Detailed metrics
    st.subheader("📋 Detailed Metrics")
    
    with st.expander("View All Metrics (P@K, R@K, NDCG@K)"):
        # Group metrics
        precision_metrics = {k: v for k, v in metrics.items() if "precision" in k}
        recall_metrics = {k: v for k, v in metrics.items() if "recall" in k}
        ndcg_metrics = {k: v for k, v in metrics.items() if "ndcg" in k}
        mrr = metrics.get("mrr", {})
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("#### Precision@K")
            for k, v in sorted(precision_metrics.items()):
                k_val = k.split("_")[1]
                st.metric(f"P@{k_val}", f"{v['local']:.4f}", f"{v['local']-v['gcp']:+.4f}")
        
        with col2:
            st.markdown("#### Recall@K")
            for k, v in sorted(recall_metrics.items()):
                k_val = k.split("_")[1]
                st.metric(f"R@{k_val}", f"{v['local']:.4f}", f"{v['local']-v['gcp']:+.4f}")
        
        with col3:
            st.markdown("#### NDCG@K")
            for k, v in sorted(ndcg_metrics.items()):
                k_val = k.split("_")[1]
                st.metric(f"N@{k_val}", f"{v['local']:.4f}", f"{v['local']-v['gcp']:+.4f}")
        
        with col4:
            st.markdown("#### Other")
            if mrr:
                st.metric("MRR", f"{mrr['local']:.4f}", f"{mrr['local']-mrr['gcp']:+.4f}")
    
    st.markdown("---")
    
    # Visual comparison
    st.subheader("📊 Visual Comparison")
    
    # Simple bar chart using metrics
    local_scores = [metrics.get(f"precision_5", {}).get("local", 0),
                   metrics.get(f"recall_5", {}).get("local", 0),
                   metrics.get(f"mrr", {}).get("local", 0),
                   metrics.get(f"ndcg_5", {}).get("local", 0)]
    
    gcp_scores = [metrics.get(f"precision_5", {}).get("gcp", 0),
                 metrics.get(f"recall_5", {}).get("gcp", 0),
                 metrics.get(f"mrr", {}).get("gcp", 0),
                 metrics.get(f"ndcg_5", {}).get("gcp", 0)]
    
    chart_data = {
        "Metric": ["Precision@5", "Recall@5", "MRR", "NDCG@5"],
        "Local": local_scores,
        "GCP Mock": gcp_scores
    }
    
    import pandas as pd
    df_chart = pd.DataFrame(chart_data)
    
    st.bar_chart(df_chart.set_index("Metric"))
    
    # Explanation
    st.info("""
    **How to Interpret:**
    - **Local Model**: Uses real semantic embeddings (all-MiniLM-L6-v2)
    - **GCP Mock**: Simulates Vertex AI with random embeddings
    
    The local model significantly outperforms the mock because it understands 
    semantic meaning, while the mock just generates random vectors.
    """)


def show_document_search():
    """Show interactive document search."""
    st.header("🔎 Document Search")
    
    documents = load_documents()
    
    if not documents:
        st.error("No documents found!")
        return
    
    st.success(f"Loaded {len(documents)} documents")
    
    # Search input
    query = st.text_input("Enter your search query:", placeholder="e.g., What is machine learning?")
    
    if query:
        st.info("Note: For actual search, you'd need to re-run the pipeline. Showing sample documents below.")
    
    # Show sample documents
    st.subheader("📄 Sample Documents")
    
    for doc in documents[:10]:
        with st.expander(f"📄 {doc.get('title', 'Untitled')}"):
            st.markdown(f"**ID:** {doc.get('id', 'N/A')}")
            st.markdown(f"**Topic:** {doc.get('topic', 'N/A')}")
            st.markdown(f"**Concept:** {doc.get('concept', 'N/A')}")
            st.markdown(f"**Words:** {doc.get('word_count', 'N/A')}")
            st.markdown("---")
            st.markdown(doc.get('text', '')[:500] + "...")


def show_eval_data():
    """Show evaluation data."""
    st.header("📋 Evaluation Data")
    
    eval_pairs = load_eval_pairs()
    
    if not eval_pairs:
        st.error("No evaluation data found!")
        return
    
    st.success(f"Loaded {len(eval_pairs)} evaluation pairs")
    
    # Show eval pairs
    st.subheader("Query-Document Pairs")
    
    for pair in eval_pairs[:10]:
        with st.expander(f"❓ {pair.get('query', 'N/A')[:80]}..."):
            st.markdown(f"**ID:** {pair.get('id', 'N/A')}")
            st.markdown(f"**Query:** {pair.get('query', 'N/A')}")
            st.markdown(f"**Relevant Docs:** {', '.join(pair.get('relevant_docs', []))}")
            st.markdown(f"**Concept:** {pair.get('concept', 'N/A')}")
            st.markdown(f"**# Relevant:** {pair.get('n_relevant', 'N/A')}")


def show_about():
    """Show about page."""
    st.header("ℹ️ About This Project")
    
    st.markdown("""
    ## RAG Vector Search Assessment
    
    This is a local Retrieval-Augmented Generation (RAG) system that demonstrates:
    
    - **Semantic Search**: Using sentence embeddings (all-MiniLM-L6-v2)
    - **Vector Store**: FAISS for fast similarity search
    - **GCP Mocking**: Simulating Vertex AI behavior without API keys
    - **Benchmarking**: Precision@K, Recall@K, MRR, NDCG@K
    
    ### Technology Stack
    
    | Component | Technology |
    |-----------|------------|
    | Language | Python 3.9+ |
    | Embeddings | all-MiniLM-L6-v2 (transformers) |
    | Vector Store | FAISS (faiss-cpu) |
    | Web UI | Streamlit |
    | Testing | pytest |
    
    ### How to Run
    
    ```bash
    # Clone the repository
    git clone https://github.com/badalsharma9929/rag-assessment.git
    cd rag-assessment
    
    # Install dependencies
    pip install -r requirements.txt
    
    # Run the assessment
    python main.py
    
    # Run the web app
    streamlit run streamlit_app.py
    ```
    
    ### Repository
    
    [GitHub: badalsharma9929/rag-assessment](https://github.com/badalsharma9929/rag-assessment)
    """)


if __name__ == "__main__":
    main()