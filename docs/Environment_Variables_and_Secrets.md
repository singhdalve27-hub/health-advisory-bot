# Environment Variables & Secrets Handling Note
**Course:** IS Professional Elective 4 — Generative AI Systems  
**Checkpoint 4 Deliverable 17 (Containerization & Deployment)**  
**Project:** DOH Health-Advisory Assistant  

---

## 1. Principles of Secure Configuration
In production generative AI systems, securing API credentials (such as Google Gemini API keys) and protecting runtime environments against accidental credential exposure is paramount. The Twelve-Factor App methodology mandates strict separation between codebase and configuration.

---

## 2. Environment Variables Configuration

| Variable Name | Required | Default | Purpose |
| :--- | :--- | :--- | :--- |
| `GEMINI_API_KEY` | **Yes** | None | Private API key for Google Gemini (`gemini-3.8-flash`) LLM generation and query contextualization. |
| `PORT` | Optional | `8501` | The network port the Streamlit web server binds to. Dynamically injected by Railway/cloud PaaS at container boot. |
| `PYTHONUNBUFFERED` | Optional | `1` | Ensures real-time stdout/stderr logging in Docker and Railway log consoles. |
| `HF_HOME` | Optional | `/home/appuser/.cache/huggingface` | Cache directory for the local sentence-transformers embedding model weights. |
| `TOKENIZERS_PARALLELISM` | Optional | `false` | Prevents fork deadlocks in multi-threaded container environments. |

---

## 3. Local Development vs. Production Deployment

### 3.1 Local Development Workflow
* Developers copy `.env.example` to `.env`:
  ```bash
  cp .env.example .env
  ```
* Secret keys are loaded using `python-dotenv` at script startup:
  ```python
  from dotenv import load_dotenv
  load_dotenv()
  ```
* `.gitignore` and `.dockerignore` strictly prevent `.env` from ever being staged or committed to Git or baked into Docker layers.

### 3.2 Production Cloud Deployment (Railway)
* **Zero Hardcoded Secrets:** No secrets exist in the Dockerfile or repository code.
* **Railway Secret Injection:**
  1. Navigate to the Railway project dashboard $\rightarrow$ **Variables** tab.
  2. Add the key: `GEMINI_API_KEY` with value: `AQ...` (your Google Gemini API Key).
  3. Railway securely encrypts this secret at rest and injects it directly into the container's environment variables at runtime.
* **Dynamic Port Binding:** Railway automatically allocates a random public port and injects the `PORT` environment variable. Our Dockerfile entrypoint cleanly binds Streamlit to this port:
  ```bash
  streamlit run src/app.py --server.port=${PORT:-8501} --server.address=0.0.0.0
  ```

---

## 4. Security Audit & Best Practices Checklist

- [x] **Repository Exclusions:** `.env` and `.env.*` are added to `.gitignore`.
- [x] **Docker Image Cleanliness:** `.dockerignore` excludes `.env`, `__pycache__`, and credentials from build context.
- [x] **Non-Root Execution:** The Dockerfile creates and switches to a dedicated unprivileged user (`appuser` with UID 1000) to mitigate container breakout vulnerabilities.
- [x] **Key Rotation Preparedness:** If an API key is compromised, changing the variable in the Railway dashboard immediately rotates the key across all instances without requiring a code rebuild.

