# ==============================================================================
# Multi-stage Dockerfile for Health-Advisory Assistant (Streamlit + LangChain RAG)
# Optimized for Railway, Render, and Local Docker Containers
# ==============================================================================

# ----------------- STAGE 1: Builder -----------------
FROM python:3.11-slim AS builder

WORKDIR /build

# Install compilation tools needed for C-extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency definition
COPY requirements.txt .

# Install CPU-only torch to keep the container lightweight (~1.5GB vs ~6GB)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt

# Pre-cache the embedding model into the builder stage
ENV HF_HOME=/build/hf_cache
RUN python -c "from transformers import AutoTokenizer, AutoModel; AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2'); AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')"

# ----------------- STAGE 2: Production Runner -----------------
FROM python:3.11-slim AS runner

WORKDIR /app

# Install minimal runtime libraries (curl for healthchecks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local
COPY --from=builder /build/hf_cache /root/.cache/huggingface

# Environment configuration
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8501 \
    HF_HOME=/root/.cache/huggingface \
    TOKENIZERS_PARALLELISM=false

# Copy application codebase and persistent data
COPY src/ /app/src/
COPY data/ /app/data/
COPY docs/ /app/docs/
COPY README.md /app/README.md

# Pre-index vector database during build
RUN python src/ingestion.py --rebuild-doh

# Create non-root user for security best practice
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app && \
    mkdir -p /home/appuser/.cache && \
    cp -r /root/.cache/huggingface /home/appuser/.cache/ && \
    chown -R appuser:appuser /home/appuser/.cache

USER appuser
ENV HF_HOME=/home/appuser/.cache/huggingface

# Expose Streamlit default port
EXPOSE 8501

# Healthcheck to verify the web service is responsive
HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8501}/_stcore/health || exit 1

# Start Streamlit dynamically binding to Railway's assigned $PORT
CMD ["sh", "-c", "streamlit run src/app.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.enableCORS=false --server.enableXsrfProtection=false --server.headless=true"]
