# Final Presentation & Defense Preparation Guide
**Course:** IS Professional Elective 4 — Generative AI Systems  
**Capstone Deliverable 20 (Final Presentation & Defense)**  
**Project:** DOH Health-Advisory Assistant  
**Team Members:**  
* **Dalve Singh** — Lead Developer (Data Pipeline & Embeddings)  
* **Gian Garcia** — AI Orchestrator (RAG Architecture, Memory & Prompting)  
* **Lonnette Oroyan** — DevOps & UI Engineer (Deployment & Streamlit Interface)  

---

## 1. Presentation Structure (10–15 Minutes Flow)

### Slide 1: Problem & Motivation (1 Slide) — *Speaker: Dalve Singh*
* **The Problem:** During communicable disease outbreaks in the Philippines (Dengue surges, floodwater Leptospirosis exposure, Rabies incidents), community residents and barangay health workers need fast, accurate guidance.
* **The Bottleneck:** Medical manuals are static 80-page PDFs or pamphlets. Finding immediate first-aid protocols (e.g., washing dog bites for 10-15 minutes with soap and water) takes too long.
* **The Solution:** A RAG-powered Health-Advisory Assistant grounded 100% in official DOH advisories, preventing hallucinations while offering interactive, multi-turn assistance.

### Slide 2: End-to-End System Architecture (1–2 Slides) — *Speaker: Gian Garcia*
* **Ingestion:** Automated pipeline parsing PDF text with PyPDF2, cleaning artifacts, and chunking with `RecursiveCharacterTextSplitter` (800 chars, 150 overlap).
* **Embeddings & Vector Store:** Dense embeddings generated using `sentence-transformers/all-MiniLM-L6-v2` (384-dimensions) and indexed in ChromaDB using Cosine distance.
* **LangChain Orchestration:** Conversational memory contextualizes user queries, ChromaDB retrieves top-4 chunks, and Gemini 3.8 Flash generates answers with strict DOH citations.
* **Guardrails:** Temperature set to 0.0 with deterministic refusal fallback string for unverified or out-of-scope queries.

### Slide 3: Live Application Demo (3–4 Minutes) — *Speaker: Lonnette Oroyan*
* Show the Streamlit Web Application running live on Railway.
* **Turn 1 (Medical Query):** *"What are the warning signs of Dengue?"* $\rightarrow$ Model outputs grounded clinical symptoms and DOH citation.
* **Turn 2 (Conversational Memory):** *"How is it transmitted and prevented?"* $\rightarrow$ Model understands "it" refers to Dengue, details *Aedes aegypti* and the 4S strategy.
* **Turn 3 (Emergency Protocol):** *"A stray dog bit my foot. What is the immediate first aid?"* $\rightarrow$ Model highlights the 10-15 minute washing protocol and cautions against *tandok*.
* **Turn 4 (Out-of-Scope Fallback):** *"What is the current price of Bitcoin?"* $\rightarrow$ Model deterministically returns: *"I don't have enough information to answer that based on the DOH manual."*
* **Turn 5 (Live Ingestion):** Upload a new health advisory PDF via the sidebar uploader and show instant vector indexing!

### Slide 4: Key Technical Decisions & Trade-Offs — *Speaker: Gian Garcia & Dalve Singh*
* **LangChain vs. LlamaIndex:** Selected LangChain for explicit control over prompt injection, custom memory contextualization, and lightweight containerization.
* **RAG vs. Fine-Tuning (PEFT):** Selected RAG over fine-tuning because public health requires non-parametric grounding, zero hallucination risk, and instant document update agility.
* **Cosine Distance vs. L2:** Selected Cosine distance to measure directional semantic similarity independent of chunk character lengths.

### Slide 5: Limitations & Future Enhancements — *Speaker: Lonnette Oroyan*
* **Current Limitations:** Limited to the communicable diseases cataloged in the DOH PDF; complex PDF tables require semi-structured pre-processing.
* **Future Roadmap:** Multilingual English/Tagalog/Cebuano query expansion; automated voice interface for rural barangay health posts; integration with DOH official surveillance APIs.

---

## 2. Answers to Sample Defense Questions (Section 6 Rubric)

### Question 1: "Why did you choose this distance metric / vector database over the alternatives?"
> **Model Answer (Dalve Singh):**  
> "We selected **Cosine Distance** ($1 - \text{cosine similarity}$) because in natural language text embeddings, document chunks vary in length. Euclidean ($L_2$) distance is sensitive to the magnitude of the embedding vector, which can penalize longer chunks even if they are semantically identical. Cosine metric evaluates the cosine of the angle between vectors, normalizing for vector magnitude and focusing purely on semantic orientation.  
>  
> For our vector database, we chose **ChromaDB** over cloud-managed alternatives like Pinecone or self-hosted Milvus/Qdrant because ChromaDB is lightweight, serverless, and can run embedded directly inside our Docker container with zero network latency. This kept our architecture self-contained, minimized deployment complexity on Railway, and eliminated recurring cloud vector database costs."

---

### Question 2: "Walk us through what happens, step-by-step, when a user submits a query?"
> **Model Answer (Gian Garcia):**  
> "When a user inputs a query in the Streamlit UI:  
> 1. **Contextualization:** `HealthAdvisoryRAG` checks the session memory. If previous conversation turns exist, Gemini 3.8 Flash reformulates the user query into a standalone search query to resolve pronouns and implicit references.  
> 2. **Embedding:** The standalone query is converted into a 384-dimensional dense vector using `LocalMiniLMEmbeddings` (`sentence-transformers/all-MiniLM-L6-v2`).  
> 3. **Retrieval:** ChromaDB executes an Approximate Nearest Neighbor search using cosine distance and retrieves the top-4 most relevant text chunks along with metadata.  
> 4. **Prompt Assembly:** LangChain constructs the generation prompt, packaging the system instruction guardrails, the retrieved DOH context chunks, recent chat turns, and the user's question.  
> 5. **Generation & Guardrail Check:** Gemini 3.8 Flash evaluates the prompt at temperature 0.0. If the context contains the answer, it outputs the grounded guidance with citations. If the query is out-of-scope or missing from the manual, it deterministically outputs our safe fallback string: *'I don't have enough information to answer that based on the DOH manual.'*  
> 6. **State Update:** The interaction is appended to conversational memory and rendered in the UI with citations."

---

### Question 3: "How does your system avoid or reduce hallucinated answers?"
> **Model Answer (Gian Garcia):**  
> "We employ a four-layer defense against hallucinations:  
> 1. **Zero Temperature ($T = 0.0$):** Eliminates stochastic generation variance, enforcing deterministic output selection.  
> 2. **Strict System Prompt Boundaries:** The prompt explicitly instructs the LLM: *'Answer using ONLY the information provided in the context below. Do not use outside medical knowledge or fabricate treatments.'*  
> 3. **The 'Safe Fallback' Constraint:** We established an exact deterministic refusal phrase: *'I don't have enough information to answer that based on the DOH manual.'* By giving the model an explicit, sanctioned escape route, we eliminate the model's inclination to guess.  
> 4. **Mandatory Source Citation:** The prompt requires the model to state the exact disease advisory or section title found in the retrieved context. If it cannot find a citation in the context, it triggers the fallback."

---

### Question 4: "What would break if your document collection grew 100x?"
> **Model Answer (Dalve Singh & Lonnette Oroyan):**  
> "If our dataset scaled 100x (from 80 pages to 8,000 pages / $\sim 11,000$ chunks):  
> 1. **Memory & Storage:** Embedded ChromaDB SQLite would experience slower query search times ($O(N)$ brute-force scaling without specialized indexing) and higher RAM utilization inside the container.  
>    * *Fix:* Migrate from embedded ChromaDB to an external, distributed vector database like **Qdrant, Milvus, or PostgreSQL with pgvector** utilizing HNSW (Hierarchical Navigable Small World) indexing.  
> 2. **Retrieval Degradation (Context Contamination):** A 100x larger corpus means more overlapping disease advisories with similar symptoms (e.g., fever, rash, headache). Dense cosine similarity alone would retrieve false positives.  
>    * *Fix:* Implement **Hybrid Search** (combining BM25 keyword matching with dense vectors) followed by a **Cross-Encoder Reranker** (such as `bge-reranker-large` or Cohere Rerank) to rescore top-25 candidates down to the top-4 most precise chunks before LLM generation.  
> 3. **Ingestion Pipeline Throughput:** Sequential CPU embedding would become a bottleneck.  
>    * *Fix:* Implement asynchronous batched embedding across GPU workers using Celery or Redis queues."

---

### Question 5: "If you had time to fine-tune a model with LoRA/QLoRA, what would you target and why?"
> **Model Answer (Gian Garcia & Dalve Singh):**  
> "We would **not** fine-tune the model to memorize medical facts, because RAG handles factual grounding far more safely and permits instant document updates.  
>  
> Instead, we would target **conversational style adaptation, terminology alignment, and query contextualization**:  
> 1. **Filipino/Tagalog Colloquialisms:** Community health users often use vernacular terms (e.g., *'kumukulo ang tiyan'*, *'nanginginig sa lagnat'*, *'kagat ng aso sa binti'*, *'namumula ang mata'*, *'tandok'*). A QLoRA adapter trained on bilingual community health consultations would bridge colloquial descriptions to clinical DOH manual terms.  
> 2. **Target Layers:** We would apply QLoRA to attention projection matrices ($W_q, W_v$) of an open foundation model (e.g., Llama-3-8B) with rank $r=16$ and scaling factor $\alpha=32$.  
> 3. **Preserving RAG:** The fine-tuned model would still operate as the reader/generator over retrieved ChromaDB chunks, marrying linguistic adaptability with non-parametric factual safety."

---

## 3. Team Member Role Responsibilities during Defense

| Teammate | Focus Area | Primary Q&A Topics |
| :--- | :--- | :--- |
| **Dalve Singh** | Lead Developer: Data Pipeline & Embeddings | PDF extraction, text cleaning, chunking strategies, vector embeddings (all-MiniLM-L6-v2), ChromaDB indexing, distance metrics. |
| **Gian Garcia** | AI Orchestrator: RAG & Prompting | LangChain orchestration, conversational memory, multi-turn query rewriting, hallucination mitigation, PEFT/LoRA trade-offs. |
| **Lonnette Oroyan** | DevOps & UI: Interface & Deployment | Streamlit UI design, multi-stage Docker build, Railway cloud deployment, environment secrets, CI/CD pipeline. |

