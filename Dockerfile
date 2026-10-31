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

# Install CPU-only torch first (avoids pulling full CUDA ~6GB image)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Pre-cache the HuggingFace embedding model weights into the build layer
ENV HF_HOME=/root/.cache/huggingface
RUN python -c "\
from transformers import AutoTokenizer, AutoModel; \
AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2'); \
AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2'); \
print('Model cached successfully')"

# ----------------- STAGE 2: Production Runner -----------------
FROM python:3.11-slim AS runner

WORKDIR /app

# Install minimal runtime libraries (curl for Railway healthchecks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy all installed Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy pre-cached HuggingFace model weights
COPY --from=builder /root/.cache/huggingface /root/.cache/huggingface

# Environment configuration
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8501 \
    HF_HOME=/root/.cache/huggingface \
    TOKENIZERS_PARALLELISM=false \
    HF_HUB_OFFLINE=1

# Copy application codebase and data
COPY src/ /app/src/
COPY data/ /app/data/
COPY docs/ /app/docs/
COPY README.md /app/README.md

# Pre-index vector database during build (runs ingestion pipeline)
RUN python src/ingestion.py --rebuild-doh

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app && \
    mkdir -p /home/appuser/.cache/huggingface && \
    cp -r /root/.cache/huggingface /home/appuser/.cache/ && \
    chown -R appuser:appuser /home/appuser/.cache

USER appuser
ENV HF_HOME=/home/appuser/.cache/huggingface

# Expose Streamlit default port
EXPOSE 8501

# Healthcheck for Railway deployment monitoring
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8501}/_stcore/health || exit 1

# Start Streamlit — Railway injects $PORT at runtime
CMD ["sh", "-c", "streamlit run src/app.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.enableCORS=false --server.enableXsrfProtection=false --server.headless=true"]
