# Container for the Streamlit app in app.py.
# Runs on Google Cloud Run (honours $PORT), or any Docker host (defaults to 7860).
FROM python:3.12-slim

# Build tools for the few deps without a pure wheel; curl for healthchecks.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Run as a non-root user (also matches Hugging Face Spaces' uid 1000).
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user . .

# Baked-in backend config. Corpus + downloads land in CITATION_DATA_DIR; on Cloud
# Run that is in-memory and lost on scale-to-zero — mount a volume there (GCS
# FUSE) and it persists. Secrets (HF_TOKEN) and model ids come in at deploy time.
ENV CITATION_DATA_DIR=/home/user/data \
    CITATION_LAYOUT_DETECTION=0 \
    CITATION_LOG_FILE=0 \
    LLM_BACKEND=openai \
    CITATION_EMBED_BACKEND=huggingface \
    PORT=7860
RUN mkdir -p /home/user/data

EXPOSE 7860
HEALTHCHECK CMD curl -f "http://localhost:${PORT}/_stcore/health" || exit 1

# Shell form so ${PORT} expands — Cloud Run sets it to 8080.
CMD streamlit run app.py \
      --server.port=${PORT} --server.address=0.0.0.0 \
      --server.headless=true --browser.gatherUsageStats=false
