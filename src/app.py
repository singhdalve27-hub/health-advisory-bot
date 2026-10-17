import os
import sys
import time
import warnings
import streamlit as st

# Suppress background library warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Add project root to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.rag_chain import HealthAdvisoryRAG, FALLBACK_REFUSAL_MESSAGE
from src.ingestion import ingest_document, get_chroma_collection

st.set_page_config(
    page_title="DOH Health-Advisory Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished government-standard health portal appearance
st.markdown("""
<style>
    .main-header {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #0d47a1;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #37474f;
        font-size: 1.05rem;
        margin-bottom: 1.2rem;
    }
    .disclaimer-box {
        background-color: #e3f2fd;
        border-left: 5px solid #1976d2;
        padding: 0.8rem 1.2rem;
        border-radius: 4px;
        margin-bottom: 1.5rem;
        font-size: 0.9rem;
        color: #0d47a1;
    }
    .refusal-badge {
        background-color: #ffebee;
        color: #c62828;
        border: 1px solid #ef9a9a;
        padding: 0.3rem 0.6rem;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        margin-top: 0.5rem;
    }
    .source-box {
        background-color: #f5f5f5;
        border-radius: 5px;
        padding: 0.75rem;
        margin-top: 0.5rem;
        border: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Initializing RAG Engine & Embedding Models...")
def load_rag_engine():
    """Cache the RAG instance in memory to ensure fast responses."""
    return HealthAdvisoryRAG()

def get_doc_count():
    try:
        col = get_chroma_collection()
        return col.count()
    except Exception:
        return 0

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "show_sources" not in st.session_state:
    st.session_state.show_sources = True
if "input_query" not in st.session_state:
    st.session_state.input_query = ""

rag = load_rag_engine()

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/cd/Department_of_Health_%28DOH%29.svg/240px-Department_of_Health_%28DOH%29.svg.png", width=110)
    st.title("System Controls")
    
    # Live Vector Store Stats
    doc_count = get_doc_count()
    st.metric(label="ChromaDB Indexed Chunks", value=doc_count)
    
    st.markdown("---")
    st.subheader("⚙️ Architecture Details")
    st.markdown("""
    - **LLM:** Gemini 3.8 Flash
    - **Orchestration:** LangChain RAG
    - **Memory:** Multi-turn Context Rephrasing
    - **Embeddings:** all-MiniLM-L6-v2 (384d)
    - **Vector DB:** ChromaDB (Cosine)
    - **Guardrails:** Safe Deterministic Fallback
    """)
    
    st.markdown("---")
    st.subheader("🧪 Demo Query Shortcuts")
    st.caption("Click any query to test grounding or safe refusal:")
    
    sample_queries = [
        ("🦟 Dengue Symptoms", "What are the common warning signs of Dengue?"),
        ("🌊 Leptospirosis First Aid", "What should I do if I was exposed to floodwaters?"),
        ("🐕 Rabies Protocol", "How should an animal bite wound be cleaned immediately?"),
        ("🤧 Measles Transmission", "How is measles spread and what is its incubation period?"),
        ("🚫 Out-of-Scope (Refusal)", "What is the current price of Bitcoin?")
    ]
    
    for label, query_text in sample_queries:
        if st.button(label, use_container_width=True):
            st.session_state.sample_to_run = query_text

    st.markdown("---")
    st.subheader("📥 Ingest New Health Advisory")
    st.caption("Automated document pipeline (Deliverable 10):")
    uploaded_file = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])
    if uploaded_file is not None:
        if st.button("Ingest Document", type="primary", use_container_width=True):
            with st.spinner(f"Parsing, chunking, embedding, and indexing '{uploaded_file.name}'..."):
                file_bytes = uploaded_file.read()
                result = ingest_document(file_bytes, filename=uploaded_file.name)
                if result.get("success"):
                    st.success(f"Successfully indexed {result['chunk_count']} new chunks in {result['elapsed_seconds']}s!")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error(f"Ingestion failed: {result.get('error')}")

    st.markdown("---")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        rag.clear_memory()
        st.rerun()
        
    st.session_state.show_sources = st.checkbox("Show Retrieved Sources", value=st.session_state.show_sources)

# ----------------- MAIN UI -----------------
st.markdown('<div class="main-header">🇵🇭 DOH Health-Advisory Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Automated Generative AI Capstone RAG Application for Communicable Diseases</div>', unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer-box">
    <strong>⚠️ Official Public Health Notice:</strong>
    This assistant is grounded strictly in official Department of Health (DOH) health advisories.
    It provides educational guidance and immediate first-aid protocols. It is <em>not</em> a substitute for professional clinical diagnosis. In case of acute medical emergencies, visit your nearest Barangay Health Station or hospital immediately.
</div>
""", unsafe_allow_html=True)

# Render Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("is_refusal"):
            st.markdown('<div class="refusal-badge">🛡️ Guardrail Active: Query Out of Scope / Unverified in DOH Manual</div>', unsafe_allow_html=True)
        if st.session_state.show_sources and msg.get("sources"):
            with st.expander("📚 Retrieved DOH Manual Sources & Chunks"):
                for idx, src in enumerate(msg["sources"]):
                    doc_src = src["metadata"].get("source", "DOH Manual")
                    chunk_id = src["metadata"].get("chunk_id", f"chunk_{idx}")
                    sim = src.get("similarity_score", "N/A")
                    st.markdown(f"**[{idx+1}] {doc_src}** (`{chunk_id}`) — Similarity: `{sim}`")
                    st.caption(src["content"])

# Process Sample Query or User Input
query_to_process = None

if "sample_to_run" in st.session_state:
    query_to_process = st.session_state.pop("sample_to_run")
else:
    user_prompt = st.chat_input("Ask a question about dengue, leptospirosis, rabies, malaria, measles...")
    if user_prompt:
        query_to_process = user_prompt

if query_to_process:
    # 1. Display User Message
    st.session_state.messages.append({"role": "user", "content": query_to_process})
    with st.chat_message("user"):
        st.markdown(query_to_process)

    # 2. Generate Assistant Response
    with st.chat_message("assistant"):
        with st.spinner("Searching DOH guidelines & retrieving grounded context..."):
            res = rag.ask(query_to_process)
            
            st.markdown(res["answer"])
            if res.get("is_refusal"):
                st.markdown('<div class="refusal-badge">🛡️ Guardrail Active: Query Out of Scope / Unverified in DOH Manual</div>', unsafe_allow_html=True)
                
            if st.session_state.show_sources and res.get("sources"):
                with st.expander("📚 Retrieved DOH Manual Sources & Chunks"):
                    for idx, src in enumerate(res["sources"]):
                        doc_src = src["metadata"].get("source", "DOH Manual")
                        chunk_id = src["metadata"].get("chunk_id", f"chunk_{idx}")
                        sim = src.get("similarity_score", "N/A")
                        st.markdown(f"**[{idx+1}] {doc_src}** (`{chunk_id}`) — Similarity: `{sim}`")
                        st.caption(src["content"])

    # 3. Save to Session State
    st.session_state.messages.append({
        "role": "assistant",
        "content": res["answer"],
        "is_refusal": res.get("is_refusal"),
        "sources": res.get("sources")
    })

