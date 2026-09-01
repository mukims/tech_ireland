# Hugging Face Spaces — Docker SDK. Runs the Streamlit app in app.py.
FROM python:3.12-slim

# Build tools for the few deps without a pure wheel; curl for healthchecks.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces runs the container as uid 1000.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user . .

# Writable data dir inside the image. If you attach HF persistent storage,
# override this with CITATION_DATA_DIR=/data in the Space variables.
ENV CITATION_DATA_DIR=/home/user/data \
    CITATION_LAYOUT_DETECTION=0 \
    LLM_BACKEND=openai \
    CITATION_EMBED_BACKEND=huggingface
RUN mkdir -p /home/user/data

# HF Spaces expects the app on the port named by `app_port` in README.md (7860).
EXPOSE 7860
HEALTHCHECK CMD curl -f http://localhost:7860/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.port=7860", "--server.address=0.0.0.0", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]
