"""
RAG Vector Search Assessment - Streamlit Web App

Clean, modern interface to visualize RAG assessment results.
"""

import streamlit as st
import json
from pathlib import Path
import pandas as pd

st.set_page_config(
    page_title="RAG Assessment Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Clean modern theme
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1.5rem 0;
    }
    .main-header h1 {
        color: #1a56db;
        font-size: 2.5rem;
        font-weight: 700;
    }
    .main-header p {
        color: #6b7280;
        font-size: 1.2rem;
    }
    .card {
        background: #ffffff;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        border: 1px solid #e5e7eb;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: #f0f9ff;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        border: 1px solid #bae6fd;
    }
    .metric-card h3 {
        color: #1a56db;
        font-size: 1rem;
        margin: 0;
    }
    .metric-card .value {
        color: #059669;
        font-size: 1.8rem;
        font-weight: 700;
    }
    .feature-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        border: 1px solid #e5e7eb;
        transition: transform 0.2s;
    }
    .feature-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .feature-card .icon {
        font-size: 2rem;
    }
    .feature-card .title {
        color: #1a56db;
        font-weight: 600;
        margin: 0.5rem 0 0.2rem;
    }
    .feature-card .desc {
        color: #6b7280;
        font-size: 0.85rem;
    }
    .stat-box {
        background: #ffffff;
        border-left: 4px solid #1a56db;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .nav-header {
        text-align: center;
        padding: 1rem 0;
        border-bottom: 1px solid #e5e7eb;
        margin-bottom: 1rem;
    }
    .nav-header h2 {
        color: #1a56db;
        margin: 0;
    }
</style>
""", unsafe_allow_html=True)

DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")
DOCS_FILE = DATA_DIR / "documents.json"
EVAL_FILE = DATA_DIR / "eval_pairs.json"
RESULTS_FILE = OUTPUT_DIR / "results.json"


@st.cache_data
def load_results():
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return None

@st.cache_data
def load_documents():
    if DOCS_FILE.exists():
        with open(DOCS_FILE) as f:
            return json.load(f)
    return []

@st.cache_data
def load_eval_pairs():
    if EVAL_FILE.exists():
        with open(EVAL_FILE) as f:
            return json.load(f)
    return []


def main():
    st.markdown("""
    <div class="main-header">
        <h1>🔍 RAG Vector Search Assessment</h1>
        <p>Local RAG System with FAISS + Semantic Embeddings | all-MiniLM-L6-v2</p>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown('<div class="nav-header"><h2>📊 Navigation</h2></div>', unsafe_allow_html=True)
        page = st.radio("", ["Home", "Results", "Search", "Data", "About"], label_visibility="collapsed")
        st.markdown("---")

        docs = load_documents()
        evals = load_eval_pairs()
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Documents", len(docs))
        with col2:
            st.metric("Queries", len(evals))

        st.markdown("---")
        st.markdown(
            '<a href="https://github.com/badalsharma9929/rag-assessment" target="_blank">'
            '<button style="background:#1a56db;color:white;border:none;border-radius:8px;'
            'padding:10px 20px;width:100%;cursor:pointer;font-weight:600;">'
            '🔗 View on GitHub</button></a>',
            unsafe_allow_html=True
        )

    pages = {"Home": show_home, "Results": show_results, "Search": show_search,
             "Data": show_data, "About": show_about}
    pages.get(page, show_home)()


def show_home():
    st.subheader("📋 System Overview")

    col1, col2, col3, col4 = st.columns(4)
    for col, icon, title, desc in zip(
        [col1, col2, col3, col4],
        ["🔍", "💾", "📊", "☁️"],
        ["Semantic Search", "FAISS Storage", "Benchmark Metrics", "GCP Mock"],
        ["Real embeddings", "Fast similarity", "P@K, R@K, MRR", "No API needed"]
    ):
        with col:
            st.markdown(f'''
            <div class="feature-card">
                <div class="icon">{icon}</div>
                <div class="title">{title}</div>
                <div class="desc">{desc}</div>
            </div>
            ''', unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("📈 Latest Results")

    results = load_results()
    if results:
        m = results.get("metrics", {})
        cols = st.columns(4)
        for col, label, key in zip(cols, ["Precision@5", "Recall@5", "MRR", "NDCG@5"],
                                     ["precision_5", "recall_5", "mrr", "ndcg_5"]):
            val = m.get(key, {}).get("local", 0)
            with col:
                st.markdown(f'<div class="metric-card"><h3>{label}</h3><div class="value">{val:.2%}</div></div>',
                            unsafe_allow_html=True)

        st.info("Navigate to **Results** for full comparison with GCP Mock.")
    else:
        st.warning("No results found. Run `python main.py` first.")


def show_results():
    st.subheader("📈 Benchmark Results")

    results = load_results()
    if not results:
        st.error("No results found. Run `python main.py` first.")
        return

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f'<div class="card"><strong>🟢 Local:</strong> {results["local_model"]}<br>'
                    f'<span style="color:#6b7280;">Real semantic embeddings</span></div>',
                    unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="card"><strong>🟠 GCP Mock:</strong> {results["gcp_model"]}<br>'
                    f'<span style="color:#6b7280;">Simulated random embeddings</span></div>',
                    unsafe_allow_html=True)

    st.markdown("### 📊 Metric Comparison")
    m = results.get("metrics", {})

    rows = []
    for name, key in [("Precision@1", "precision_1"), ("Precision@3", "precision_3"),
                       ("Precision@5", "precision_5"), ("Recall@1", "recall_1"),
                       ("Recall@3", "recall_3"), ("Recall@5", "recall_5"),
                       ("MRR", "mrr"), ("NDCG@5", "ndcg_5")]:
        if key in m:
            rows.append({"Metric": name, "Local": f"{m[key]['local']:.4f}",
                        "GCP Mock": f"{m[key]['gcp']:.4f}",
                        "Difference": f"{m[key]['local']-m[key]['gcp']:+.4f}"})
    if rows:
        st.dataframe(pd.DataFrame(rows).set_index("Metric"), use_container_width=True)

    st.markdown("### 📊 Visual Comparison")
    chart = pd.DataFrame({
        "Metric": ["Precision@5", "Recall@5", "MRR", "NDCG@5"],
        "Local": [m.get("precision_5", {}).get("local", 0),
                  m.get("recall_5", {}).get("local", 0),
                  m.get("mrr", {}).get("local", 0),
                  m.get("ndcg_5", {}).get("local", 0)],
        "GCP Mock": [m.get("precision_5", {}).get("gcp", 0),
                     m.get("recall_5", {}).get("gcp", 0),
                     m.get("mrr", {}).get("gcp", 0),
                     m.get("ndcg_5", {}).get("gcp", 0)]
    }).set_index("Metric")
    st.bar_chart(chart)

    with st.expander("View All Metrics (all K values)"):
        all_rows = []
        for metric, vals in sorted(m.items()):
            all_rows.append({"Metric": metric, "Local": f"{vals['local']:.4f}",
                            "GCP": f"{vals['gcp']:.4f}",
                            "Diff": f"{vals['local']-vals['gcp']:+.4f}"})
        if all_rows:
            st.table(pd.DataFrame(all_rows).set_index("Metric"))

    st.success("Local model outperforms GCP mock because it uses real semantic embeddings. The mock generates random vectors based on text hash.")


def show_search():
    st.subheader("🔎 Document Explorer")

    documents = load_documents()
    if not documents:
        st.error("No documents found.")
        return

    st.success(f"Loaded **{len(documents)}** documents")

    search = st.text_input("Search by title or concept:", placeholder="e.g., neural networks")
    filtered = documents
    if search:
        filtered = [d for d in documents if search.lower() in d.get("title", "").lower()
                    or search.lower() in d.get("concept", "").lower()]

    st.caption(f"Showing {len(filtered)} documents")
    for doc in filtered:
        with st.expander(f"📄 {doc.get('title', 'Untitled')}"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**ID:** `{doc.get('id')}`")
                st.markdown(f"**Topic:** {doc.get('topic')}")
            with col2:
                st.markdown(f"**Concept:** {doc.get('concept')}")
                st.markdown(f"**Words:** {doc.get('word_count')}")
            st.markdown("---")
            st.markdown(doc.get("text", "")[:600] + "...")


def show_data():
    st.subheader("📋 Evaluation Data")

    pairs = load_eval_pairs()
    if not pairs:
        st.error("No evaluation data found.")
        return

    total_rel = sum(p.get("n_relevant", 0) for p in pairs)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Pairs", len(pairs))
    with col2:
        st.metric("Total Relevant", total_rel)
    with col3:
        st.metric("Avg Relevant/Pair", f"{total_rel/len(pairs):.1f}")

    for pair in pairs:
        with st.expander(f"❓ {pair.get('query', '')[:70]}..."):
            st.markdown(f"**Query:** {pair.get('query')}")
            st.markdown(f"**Relevant Docs:** `{', '.join(pair.get('relevant_docs', []))}`")
            st.markdown(f"**Concept:** {pair.get('concept')}")


def show_about():
    st.subheader("ℹ️ About This Project")

    st.markdown(f"""
    <div class="card">
        <h3 style="color:#1a56db;">🔍 RAG Vector Search Assessment</h3>
        <p>Local RAG system demonstrating semantic search with FAISS + sentence embeddings.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Features")
    for icon, title, desc in [
        ("🔍", "Semantic Search", "all-MiniLM-L6-v2 embeddings"),
        ("💾", "FAISS Vector Store", "Fast similarity search"),
        ("☁️", "GCP Mock", "Simulates Vertex AI"),
        ("📊", "Benchmark", "P@K, R@K, MRR, NDCG"),
    ]:
        st.markdown(f'<div class="stat-box"><strong>{icon} {title}</strong> — {desc}</div>',
                    unsafe_allow_html=True)

    st.markdown("### Tech Stack")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Core:** Python 3.9+, Transformers, FAISS, NumPy")
    with col2:
        st.markdown("**Web:** Streamlit, Pandas, Tabulate, Pytest")

    st.markdown("### Run Commands")
    st.code("git clone https://github.com/badalsharma9929/rag-assessment.git\n"
            "cd rag-assessment\npip install -r requirements.txt\npython main.py\nstreamlit run streamlit_app.py",
            language="bash")

    st.markdown("---")
    st.success("📂 [GitHub Repository](https://github.com/badalsharma9929/rag-assessment)")


if __name__ == "__main__":
    main()