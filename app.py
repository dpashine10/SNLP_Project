"""
Streamlit Web Application: Research Discovery & Collaboration Engine.
AI University NLP Course Project (Deliverable Week 3: Version 1 Core NLP Pipeline).
"""

import sys
import time
from pathlib import Path
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import ResearchDiscoveryPipeline

# Page setup
st.set_page_config(
    page_title="Research Discovery & Collaboration Engine",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .researcher-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        transition: transform 0.15s ease-in-out, box-shadow 0.15s ease-in-out;
    }
    .researcher-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.06);
        border-color: #CBD5E1;
    }
    .badge {
        display: inline-block;
        background-color: #EEF2FF;
        color: #4F46E5;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 0.4rem;
        margin-top: 0.3rem;
    }
    .score-badge {
        background-color: #ECFDF5;
        color: #059669;
        font-weight: 700;
        font-size: 0.9rem;
        padding: 0.25rem 0.6rem;
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading Core NLP Pipeline & Embeddings...")
def get_pipeline():
    return ResearchDiscoveryPipeline()


def main():
    st.markdown('<div class="main-title">🎓 Research Discovery & Collaboration Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">AI University NLP Course Project — Version 1.0 Core NLP Pipeline (Sentence Transformers vs TF-IDF)</div>', unsafe_allow_html=True)

    pipeline = get_pipeline()

    # Sidebar settings
    st.sidebar.header("⚙️ Search Configuration")
    mode = st.sidebar.selectbox(
        "Retrieval Architecture",
        options=["Semantic (Sentence-BERT)", "TF-IDF (Keyword Baseline)", "Hybrid (Semantic + TF-IDF)"],
        index=0,
        help="Select between Dense Semantic Embeddings, Traditional Sparse TF-IDF, or Hybrid Blending."
    )

    mode_key_map = {
        "Semantic (Sentence-BERT)": "semantic",
        "TF-IDF (Keyword Baseline)": "tfidf",
        "Hybrid (Semantic + TF-IDF)": "hybrid"
    }
    active_mode = mode_key_map[mode]

    top_k = st.sidebar.slider("Number of Top Researchers", min_value=5, max_value=25, value=10, step=5)

    alpha = 0.7
    if active_mode == "hybrid":
        alpha = st.sidebar.slider(
            "Semantic Weight (Alpha)",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05,
            help="1.0 = 100% Dense Semantic, 0.0 = 100% Sparse TF-IDF"
        )

    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Dataset Statistics")
    if pipeline.profiles_df is not None:
        st.sidebar.metric("Faculty & Researchers", f"{len(pipeline.profiles_df):,}")
    if pipeline.papers_df is not None:
        st.sidebar.metric("Indexed Academic Papers", f"{len(pipeline.papers_df):,}")
    st.sidebar.metric("Embedding Model", "all-MiniLM-L6-v2")

    # Tabs
    tab_search, tab_compare, tab_eval = st.tabs(["🔍 Research Search", "⚖️ Model Comparison", "📈 Benchmark & Metrics"])

    with tab_search:
        # Example queries
        st.write("**Quick Query Presets:**")
        cols = st.columns(4)
        preset_query = None
        if cols[0].button("NLP & Information Extraction"):
            preset_query = "Application of Large Language Models to information extraction"
        if cols[1].button("Transformer Classification"):
            preset_query = "transformer based text classification"
        if cols[2].button("Knowledge Graph Reasoning"):
            preset_query = "knowledge graph embeddings and reasoning"
        if cols[3].button("Cross-Lingual Retrieval"):
            preset_query = "cross-lingual information retrieval and neural search"

        query_input = st.text_area(
            "Enter research topic, problem description, or paper abstract:",
            value=preset_query if preset_query else "",
            placeholder="e.g. Seeking collaborators working on deep learning for biomedical text summarization and named entity recognition...",
            height=100
        )

        search_btn = st.button("🔎 Find Relevant Researchers", type="primary", use_container_width=True)

        if search_btn and query_input.strip():
            start_time = time.time()
            results = pipeline.search(
                query=query_input.strip(),
                top_k=top_k,
                mode=active_mode,
                alpha=alpha
            )
            elapsed = time.time() - start_time

            st.success(f"Found {len(results)} researchers matching query using **{mode}** in {elapsed:.3f}s")

            for res in results:
                paper = res["relevant_paper"]
                topics = res["related_topics"]

                with st.container():
                    st.markdown(f"""
                    <div class="researcher-card">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                            <span style="font-size: 1.25rem; font-weight: 700; color: #0F172A;">
                                #{res['rank']} {res['author_name']}
                            </span>
                            <span class="score-badge">Similarity: {res['similarity_score']:.4f}</span>
                        </div>
                        <div style="font-size: 0.9rem; color: #475569; margin-bottom: 0.6rem;">
                            <strong>Total Indexed Papers:</strong> {res['n_papers']} &nbsp;|&nbsp; 
                            <strong>Author ID:</strong> <a href="{res['author_id']}" target="_blank">{res['author_id']}</a>
                        </div>
                        <div style="background: #FFFFFF; border-left: 4px solid #4F46E5; padding: 0.6rem 0.8rem; border-radius: 4px; margin-bottom: 0.6rem;">
                            <strong style="color: #1E293B;">Most Relevant Paper:</strong> <em>"{paper['title']}"</em> ({paper.get('year', 'N/A')})
                            <br><span style="font-size: 0.82rem; color: #64748B;">Paper Match Score: {paper.get('paper_score', 0.0):.4f}</span>
                        </div>
                        <div>
                            <strong style="font-size: 0.85rem; color: #334155;">Key Research Topics:</strong><br>
                            {''.join([f'<span class="badge">{t}</span>' for t in topics]) if topics else '<span style="font-size:0.85rem;color:#94A3B8;">None listed</span>'}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

    with tab_compare:
        st.subheader("Side-by-Side Model Comparison")
        st.write("Compare how **Sentence Transformers (Dense Semantic Similarity)** captures conceptual relevance compared to **TF-IDF (Keyword-based)**.")

        comp_query = st.text_input(
            "Comparison Query:",
            value="transformer based text classification",
            key="comp_query_input"
        )

        if st.button("Compare Models", key="btn_compare"):
            col_sem, col_tfidf = st.columns(2)

            with col_sem:
                st.markdown("### 🧠 Sentence Transformers (Dense Semantic)")
                res_sem = pipeline.search(query=comp_query, top_k=5, mode="semantic")
                for r in res_sem:
                    st.markdown(f"**#{r['rank']} {r['author_name']}** (Score: `{r['similarity_score']:.4f}`)")
                    st.caption(f"Relevant Paper: *{r['relevant_paper']['title']}*")
                    st.markdown("---")

            with col_tfidf:
                st.markdown("### 🔤 TF-IDF Baseline (Sparse Keyword)")
                res_tfidf = pipeline.search(query=comp_query, top_k=5, mode="tfidf")
                for r in res_tfidf:
                    st.markdown(f"**#{r['rank']} {r['author_name']}** (Score: `{r['similarity_score']:.4f}`)")
                    st.caption(f"Relevant Paper: *{r['relevant_paper']['title']}*")
                    st.markdown("---")

    with tab_eval:
        st.subheader("Benchmark & Quantitative Evaluation")
        st.write("Evaluation results on the ground-truth labeled benchmark queries comparing **TF-IDF Baseline** against **Sentence Transformers**.")

        comp_file = PROJECT_ROOT / "Data" / "processed" / "calculated" / "model_comparison.csv"
        sem_file = PROJECT_ROOT / "Data" / "processed" / "calculated" / "semantic_metrics.csv"
        base_file = PROJECT_ROOT / "Data" / "processed" / "calculated" / "baseline_metrics.csv"

        if comp_file.exists():
            df_comp = pd.read_csv(comp_file)
            st.markdown("#### Overall Model Performance Summary")
            st.dataframe(df_comp, use_container_width=True)

        col_b, col_s = st.columns(2)
        with col_b:
            if base_file.exists():
                st.markdown("#### TF-IDF Baseline by Query")
                st.dataframe(pd.read_csv(base_file), use_container_width=True)

        with col_s:
            if sem_file.exists():
                st.markdown("#### Sentence Transformers by Query")
                st.dataframe(pd.read_csv(sem_file), use_container_width=True)


if __name__ == "__main__":
    main()
