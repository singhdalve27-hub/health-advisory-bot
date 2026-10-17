# Framework Comparison Note: LangChain vs. LlamaIndex
**Course:** IS Professional Elective 4 — Generative AI Systems  
**Checkpoint 3 Deliverable (Semi-Final)**  
**Project:** DOH Health-Advisory Assistant  
**Authors:** Dalve Singh, Gian Garcia, Lonnette Oroyan  

---

## 1. Executive Summary & Objective
Modern Retrieval-Augmented Generation (RAG) frameworks abstract complex vector workflows, document transformations, retrieval strategies, and prompt chains into unified pipelines. For our capstone project—the **DOH Health-Advisory Assistant**—the primary requirement is delivering clinical-standard grounding with conversational multi-turn memory, zero-hallucination guardrails, and deterministic fallback responses for out-of-scope medical inquiries.

This note provides a technical comparison of how **LangChain** and **LlamaIndex** approach **data routing, node indexing, and orchestration**, and outlines our architectural rationale for selecting specific framework components.

---

## 2. Core Architectural Differences

| Architectural Dimension | LangChain | LlamaIndex |
| :--- | :--- | :--- |
| **Primary Philosophy** | General-purpose modular orchestration and execution graph (chains, agents, memory). | Specialized data framework optimized for data ingestion, indexing, and structured retrieval. |
| **Data Unit Abstraction** | `Document` (raw page content + metadata dictionary). | `Node` (first-class chunk containing semantic relationships, preceding/succeeding node pointers, and hierarchical embeddings). |
| **Indexing Paradigm** | Vector Store wrappers (`Chroma`, `FAISS`, `PGVector`) acting as passive storage targets. | Semantic Index hierarchies (`VectorStoreIndex`, `SummaryIndex`, `TreeIndex`, `KnowledgeGraphIndex`). |
| **Data Routing & Querying** | Explicit LCEL (LangChain Expression Language) composable chains, RouterRunnables, and conversational wrappers. | Dynamic Query Engines, Sub-Question Query Engines, and Router Query Engines via LLM-based selector mechanisms. |
| **State & Memory** | First-class session state management (`BaseChatMessageHistory`, multi-turn message contextualization). | Chat engines (`CondenseQuestionChatEngine`, `ContextChatEngine`) with built-in buffer memory. |

---

## 3. Data Routing & Node Indexing Deep-Dive

### 3.1 Node Indexing & Semantic Relationships
* **LlamaIndex Approach:** In LlamaIndex, documents are divided into `Node` objects. Each node retains contextual awareness of neighboring chunks (`node.prev_node`, `node.next_node`) and parent documents (`node.source_node`). This enables sophisticated indexing structures such as *Hierarchical Indexing* (sentence window retrieval, auto-merging parent retrievers).
* **LangChain Approach:** LangChain treats data primarily through text-splitters (e.g., `RecursiveCharacterTextSplitter`) yielding lightweight `Document` instances. Indexing is decoupled from retrieval: vector stores (such as ChromaDB) store embeddings, while LangChain provides modular `VectorStoreRetriever` interfaces with cosine, L2, or maximal marginal relevance (MMR) strategies.

### 3.2 Data Routing & Chain Composition
* **LlamaIndex Approach:** Query routing is often handled declaratively via `RouterQueryEngine` with `LLMSingleSelector` or `RouterComponent`. The framework automatically classifies incoming queries to route between a vector index (for factual lookup) and a summary index (for high-level summaries).
* **LangChain Approach:** Routing is constructed using LCEL (LangChain Expression Language) or conditional branches. Chaining is highly customizable, allowing fine-grained control over prompt injection, context formatting, guardrail validation, and multi-turn query reformulation.

---

## 4. Implementation Choice & Selected Components

For our production deployment on Railway, we selected **LangChain** as our core orchestration framework, paired with **ChromaDB** for vector storage and **Gemini 3.8 Flash** via `ChatGoogleGenerativeAI`.

### Components Used:
1. **`langchain_text_splitters.RecursiveCharacterTextSplitter`:**
   * Used in our automated ingestion pipeline (`src/ingestion.py`).
   * Splits dense medical PDFs into 800-character chunks with a 150-character overlap using paragraph and sentence boundaries (`["\n\n", "\n", ". ", " "]`). This guarantees that multi-step first aid instructions (e.g., animal bite wound irrigation) are not truncated mid-sentence.
2. **`langchain_core.embeddings.Embeddings` Interface:**
   * Implemented in `src/embeddings.py` as `LocalMiniLMEmbeddings`.
   * Standardizes dense vector generation using `sentence-transformers/all-MiniLM-L6-v2` with PyTorch mean pooling, ensuring identical embeddings across ingestion and inference.
3. **Conversational Memory via Query Contextualization:**
   * Multi-turn dialogue is maintained through a two-step conversational RAG pattern.
   * If chat history exists, a reformulator LLM resolves pronouns and conversational references into a standalone query (e.g., Turn 1: *"What causes Leptospirosis?"* $\rightarrow$ Turn 2: *"What are its warning signs?"* is rewritten to *"What are the warning signs of Leptospirosis?"*).
4. **Deterministic Safe Fallback & System Guardrails:**
   * Embedded within LangChain's `ChatPromptTemplate` to enforce zero outside medical knowledge and trigger the exact deterministic fallback message when confidence is low or information is absent from the DOH manual.

### Why LangChain was Chosen:
* **Customizability over Black-Box Engines:** While LlamaIndex offers high-level query engines, LangChain's pipeline gave us direct control over the exact prompt formatting, system instructions, and deterministic fallback checking required for medical-adjacent public health safety.
* **Lightweight Container Overhead:** LangChain's modular decoupling allowed us to package only the necessary core components, resulting in faster startup times and lower RAM consumption when deploying within Railway's constrained container environment.

