# 🇵🇭 DOH Health-Advisory Assistant
> **A Production-Grade Generative AI RAG Application for Community Health & Communicable Diseases**  
> *IS Professional Elective 4 — Gen AI Capstone Project (Semestral Checkpoints 3 & 4)*

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B.svg)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-Orchestration-00A67E.svg)](https://www.langchain.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Storage-orange.svg)](https://www.trychroma.com/)
[![Gemini](https://img.shields.io/badge/Model-Gemini_3.8_Flash-4285F4.svg)](https://ai.google.dev/)
[![Docker](https://img.shields.io/badge/Docker-Multi--stage_Build-2496ED.svg)](https://www.docker.com/)
[![Railway](https://img.shields.io/badge/Deploy-Railway-0B0D0E.svg)](https://railway.app/)

---

## 📌 Project Overview
The **DOH Health-Advisory Assistant** is an intelligent, conversational Retrieval-Augmented Generation (RAG) system engineered to provide accurate, verified, and plain-language public health guidance during disease outbreaks in the Philippines (e.g., Dengue, Leptospirosis, Rabies, Measles). 

Grounded strictly in the official **Department of Health (DOH) Manual on Communicable Diseases** (`7323175-Doh-Communicable-Diseases.pdf`), the assistant guarantees:
1. **Zero External Hallucinations:** Answers are generated using *only* retrieved DOH context.
2. **Conversational Multi-Turn Memory:** Resolves follow-up queries and conversational references across multiple turns.
3. **Deterministic Out-of-Scope Fallback:** Unverified medical questions or non-health inquiries automatically trigger an explicit safe refusal string.
4. **Automated Ingestion Pipeline:** Allows healthcare workers to upload and index new health advisories with minimal manual steps.
5. **Turnkey Cloud Deployment:** Packaged with a multi-stage Docker container optimized for one-click deployment on **Railway**.

---

## 🏗️ System Architecture

```
+----------------------------------------------------------------------------------------------------+
|                                      ARCHITECTURE PIPELINE FLOW                                    |
+----------------------------------------------------------------------------------------------------+

 [User Query / Multi-turn Dialogue]
                 |
                 v
 +----------------------------------+
 |  Streamlit Web UI (src/app.py)   | <---- Health Worker Uploads New Advisory (PDF/TXT)
 +----------------------------------+          |
                 |                             v
                 v                   +----------------------------------+
 +----------------------------------+ | Automated Pipeline (ingestion.py)|
 | LangChain RAG (src/rag_chain.py) | +----------------------------------+
 |                                  |          | PyPDF2 -> RecursiveSplitter -> Embeddings
 | 1. Memory Contextualizer (LLM)   |          v
 |    Rephrases follow-up queries   | +----------------------------------+
 | 2. Dense Vector Retrieval (k=4)  | | ChromaDB Vector Store            |
 |    (all-MiniLM-L6-v2 embeddings) | | (data/chroma_db)                 |
 | 3. Grounded Prompt Assembly      | +----------------------------------+
 |    Strict Guardrails + Citation  |                  |
 +----------------------------------+                  | Retrieved Context Chunks
                 | <-----------------------------------+
                 v
 +----------------------------------+
 | Google Gemini 3.8 Flash (T=0.0)  |
 +----------------------------------+
                 |
        +--------+--------+
        |                 |
        v (Grounded)      v (Out of Scope / Missing)
 +-----------------+ +-----------------------------------------------------------+
 | Verified Advice | | Deterministic Safe Fallback:                              |
 | & Citations     | | "I don't have enough information to answer that based on  |
 +-----------------+ |  the DOH manual."                                         |
                     +-----------------------------------------------------------+
```

---

## 🗂️ Repository Structure

```text
health-advisory-bot/
├── .github/
│   └── workflows/
│       └── deploy.yml            # CI/CD automated validation & Docker build
├── data/
│   ├── 7323175-Doh-Communicable-Diseases.pdf # Official 80-page DOH manual
│   ├── cleaned_doh_manual.txt    # Cleaned extracted corpus
│   └── chroma_db/                # Persistent ChromaDB vector database
├── docs/
│   ├── Project_Proposal.md       # Capstone Project Proposal (PT 1)
│   ├── Prompt_Engineering_Report.md # Prompt guardrails & safety specs (PT 2)
│   ├── Framework_Comparison_Note.md # LangChain vs LlamaIndex analysis (Deliverable 13)
│   ├── Demo_Queries_and_Evaluation.md # 5 realistic queries + out-of-scope evaluation (Deliverable 14)
│   ├── Model_Customization_Note.md # PEFT / LoRA / QLoRA technical note (Deliverable 15)
│   ├── Environment_Variables_and_Secrets.md # Secrets handling documentation (Deliverable 17)
│   ├── Final_Technical_Report.md # Full 3-5 page consolidated report (Deliverable 19)
│   └── Final_Defense_Guide.md    # Presentation script & sample defense Q&A (Deliverable 20)
├── src/
│   ├── app.py                    # Streamlit interactive web interface (Deliverable 16)
│   ├── embeddings.py             # LangChain-compatible singleton LocalMiniLMEmbeddings
│   ├── ingestion.py              # Automated document ingestion pipeline (Deliverable 10)
│   ├── preprocess.py             # Baseline text extractor
│   ├── rag_chain.py              # LangChain RAG engine with conversational memory (Deliverables 11 & 12)
│   └── rag_demo.py               # CLI conversational interface
├── .dockerignore                 # Excludes .env, git, and caches from Docker build
├── .env.example                  # Environment variable configuration template
├── Dockerfile                    # Multi-stage production container build (Deliverable 17)
├── docker-compose.yml            # Local container orchestration
├── Procfile                      # PaaS web process definition
├── railway.json                  # Railway cloud deployment configuration
├── requirements.txt              # Production dependency specifications
└── README.md                     # System documentation & setup guide
```

---

## 🚀 Quickstart: Running Locally

### 1. Prerequisites
* Python 3.10+ installed
* Google Gemini API Key ([Get one here from Google AI Studio](https://aistudio.google.com/app/apikey))

### 2. Clone & Setup Virtual Environment
```bash
git clone https://github.com/singhdalve27-hub/health-advisory-bot.git
cd health-advisory-bot

python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the root directory:
```env
GEMINI_API_KEY=your_actual_gemini_api_key_here
PORT=8501
```

### 5. Launch the Application

#### Option A: Web Interface (Streamlit)
```bash
streamlit run src/app.py
```
Open your browser at `http://localhost:8501`.

#### Option B: Terminal CLI Interface
```bash
python src/rag_demo.py
```

#### Option C: Ingestion Pipeline CLI
```bash
# View vector database status:
python src/ingestion.py --stats

# Ingest a new PDF or text document:
python src/ingestion.py --file path/to/advisory.pdf
```

---

## 🐳 Running with Docker Locally

The project includes a multi-stage Docker build that caches embedding weights and uses CPU-only PyTorch to minimize image footprint (~1.2GB vs ~6GB):

```bash
# Build the Docker image
docker build -t health-advisory-bot:latest .

# Run the container with your API key
docker run -p 8501:8501 -e GEMINI_API_KEY="your_api_key_here" health-advisory-bot:latest
```
Or run using Docker Compose:
```bash
docker compose up --build
```

---

## ☁️ Deploying to Railway (Step-by-Step)

Deploying this application to **Railway** takes under 3 minutes:

1. **Push your code to GitHub:**
   ```bash
   git add .
   git commit -m "feat: complete Checkpoints 3 and 4 RAG application"
   git push origin main
   ```
2. **Log into Railway:**
   * Go to [railway.app](https://railway.app/) and sign in with GitHub.
3. **Create New Project:**
   * Click **New Project** $\rightarrow$ **Deploy from GitHub repo**.
   * Select `singhdalve27-hub/health-advisory-bot`.
4. **Configure Secrets / Environment Variables:**
   * In your Railway project, click on the service $\rightarrow$ navigate to the **Variables** tab.
   * Click **Add Variable** and enter:
     * **Variable Name:** `GEMINI_API_KEY`
     * **Value:** *(Paste your Google Gemini API key)*
5. **Automatic Build & Deploy:**
   * Railway automatically detects the multi-stage `Dockerfile` and `railway.json`.
   * It builds the container and maps the dynamic `$PORT`.
6. **Generate Public Domain:**
   * Under the **Settings** tab $\rightarrow$ **Networking**, click **Generate Domain**.
   * Your app is now live at: `https://health-advisory-bot-production-xxxx.up.railway.app`! 🎉

---

## 🧪 Demonstration Queries (Checkpoint 3 & 4 Rubric)

| Query Type | Sample Prompt | Expected Behavior |
| :--- | :--- | :--- |
| **Grounded Health Query** | *"What are the warning signs of Dengue?"* | Retrieves DOH manual sections; lists exact clinical warning signs and cites DOH advisory. |
| **Conversational Memory** | *(Follow-up)*: *"How is it transmitted and prevented?"* | Resolves "it" to Dengue; describes *Aedes aegypti* and the 4S strategy. |
| **Emergency First Aid** | *"A stray dog bit my foot. What is the immediate first aid?"* | Emphasizes the mandatory 10–15 minute wound washing protocol and warns against *tandok*. |
| **Flood Exposure** | *"I walked through floodwaters. What should I do for Leptospirosis?"* | Advises immediate washing and doctor consultation for post-exposure prophylaxis. |
| **Safe Refusal (Out-of-Scope)** | *"What is the current price of Bitcoin?"* | Exact safe fallback triggered: *"I don't have enough information to answer that based on the DOH manual."* |

---

## 👥 Team & Pipeline Ownership

As per the Capstone Project guidelines, each team member owns a specific component of the pipeline:

| Team Member | Role | Pipeline Component Ownership |
| :--- | :--- | :--- |
| **Dalve Singh** | Lead Developer | Data Ingestion Pipeline, Preprocessing, Tokenization, Dense Vector Embeddings (`all-MiniLM-L6-v2`), ChromaDB Storage. |
| **Gian Garcia** | AI Orchestrator | LangChain RAG Architecture, Conversational Memory, Query Contextualizer, Prompt Engineering & Hallucination Guardrails. |
| **Lonnette Oroyan** | DevOps & UI Engineer | Streamlit Web Portal, Multi-Stage Containerization (`Dockerfile`), CI/CD Pipeline, Railway Cloud Deployment. |

---

## 📄 Documentation Directory (`/docs`)
All formal course deliverable documents are consolidated in `/docs`:
* [`Framework_Comparison_Note.md`](file:///c:/Users/owner/Downloads/health-advisory-bot/docs/Framework_Comparison_Note.md) — Technical comparison of LangChain vs LlamaIndex (Deliverable 13)
* [`Demo_Queries_and_Evaluation.md`](file:///c:/Users/owner/Downloads/health-advisory-bot/docs/Demo_Queries_and_Evaluation.md) — Live demo script & test evaluation matrix (Deliverable 14)
* [`Model_Customization_Note.md`](file:///c:/Users/owner/Downloads/health-advisory-bot/docs/Model_Customization_Note.md) — PEFT, LoRA & QLoRA technical analysis (Deliverable 15)
* [`Environment_Variables_and_Secrets.md`](file:///c:/Users/owner/Downloads/health-advisory-bot/docs/Environment_Variables_and_Secrets.md) — Secrets management & Twelve-Factor configuration (Deliverable 17)
* [`Final_Technical_Report.md`](file:///c:/Users/owner/Downloads/health-advisory-bot/docs/Final_Technical_Report.md) — 3–5 page comprehensive technical documentation (Deliverable 19)
* [`Final_Defense_Guide.md`](file:///c:/Users/owner/Downloads/health-advisory-bot/docs/Final_Defense_Guide.md) — Slide deck presentation flow & defense Q&A answers (Deliverable 20)

---
*Developed for IS Professional Elective 4 — Generative AI Systems.*