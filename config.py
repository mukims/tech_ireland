"""
Central configuration for the Citation Agent pipeline.

All hardcoded model names, paths, and tunable constants live here so that
changing a model or path only requires editing one file.
"""

import os

# ─── Project Root / Data Dir ─────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Everything the pipeline *writes* (the vector store, downloaded PDFs, the JSON
# manifests, logs) is anchored here rather than to the code directory. On a
# read-only or ephemeral deployment (Hugging Face Spaces) set CITATION_DATA_DIR
# to a writable, ideally persistent, path such as /data.
DATA_DIR = os.environ.get("CITATION_DATA_DIR", PROJECT_ROOT)

# ─── LLM / Embedding Backend ─────────────────────────────────────────────────
# "ollama" (default) talks to a local Ollama daemon. "openai" talks to any
# OpenAI-compatible chat endpoint — set OPENAI_BASE_URL / OPENAI_API_KEY. This
# is what the Hugging Face Space uses (base URL = the HF router).
LLM_BACKEND     = os.environ.get("LLM_BACKEND", "ollama").lower()
# Embeddings can use a different provider from chat (e.g. hosted chat +
# HF-hosted embeddings). Defaults to whatever LLM_BACKEND is; "huggingface"
# uses huggingface_hub.InferenceClient feature-extraction.
EMBED_BACKEND   = os.environ.get("CITATION_EMBED_BACKEND", LLM_BACKEND).lower()

OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://router.huggingface.co/v1")
OPENAI_API_KEY  = os.environ.get("OPENAI_API_KEY") or os.environ.get("HF_TOKEN")
HF_TOKEN        = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY")

# ─── Models ──────────────────────────────────────────────────────────────────
# Defaults are Ollama tags. Override every one by env when pointing at a hosted
# backend, e.g. CITATION_LLM_MODEL=meta-llama/Llama-3.1-8B-Instruct.
# (gemma4:latest is not a published Ollama tag — the registry 404s on it —
#  gemma4:e2b is the variant that actually pulls.)
LLM_MODEL       = os.environ.get("CITATION_LLM_MODEL", "gemma4:e2b")
CHAT_MODEL      = os.environ.get("CITATION_CHAT_MODEL", "qwen2.5:7b")
EMBED_MODEL     = os.environ.get("CITATION_EMBED_MODEL", "nomic-embed-text")
EVAL_MODEL      = os.environ.get("CITATION_EVAL_MODEL", "deepseek-r1:14b")

# ─── Chat Model Runtime Options ──────────────────────────────────────────────
# Flash Attention + 8-bit quantized KV cache for lower latency on CPU
CHAT_OLLAMA_OPTIONS = {
    "num_ctx":    4096,        # Smaller context window = faster inference
    "num_thread": 16,          # Use most of the available CPU threads
    "flash_attn": True,        # Enable Flash Attention
    "kv_cache_type": "q8_0",   # 8-bit quantized KV cache
}

# ─── Vector Database ──────────────────────────────────────────────────────────
VECTORDB_PATH    = os.path.join(DATA_DIR, "physics_vectordb")
COLLECTION_NAME  = "physics_papers"       # detail chunks (stage-2 retrieval)
SUMMARY_COLLECTION_NAME = "physics_summaries"   # one summary per document (stage 1)
BM25_INDEX_PATH  = os.path.join(DATA_DIR, "bm25_index.pkl")

# ─── Directories ──────────────────────────────────────────────────────────────
RAW_DIR          = os.path.join(DATA_DIR, "raw")
DRAFTS_DIR       = os.path.join(DATA_DIR, "drafts")
PULLED_PDFS_DIR  = os.path.join(DATA_DIR, "pulled_pdfs")
# Figure/table crops written during ingestion. Each crop is handed straight to
# the VLM and never read back — only the generated description enters the
# corpus — so this is a debugging artefact, not corpus data, and is safe to
# delete between runs. It previously resolved to ../extracted_data/images, a
# sibling of the project, which put the output outside the repo, outside
# version control and outside any backup taken of it.
IMAGES_DIR       = os.environ.get(
    "CITATION_IMAGES_DIR", os.path.join(DATA_DIR, "images")
)

# ─── Data Files ───────────────────────────────────────────────────────────────
EXTRACTED_CITATIONS_PATH = os.path.join(DATA_DIR, "extracted_citations.json")
DOWNLOADED_JSON_PATH     = os.path.join(DATA_DIR, "downloaded.json")
FAILED_DOWNLOADS_PATH    = os.path.join(DATA_DIR, "failed_downloads.json")
# Agent 0's manifest: one record per research query it has seeded from, keyed
# by the query string. Same crash-safe rewrite-after-each-write pattern as
# downloaded.json.
SEED_PAPERS_PATH         = os.path.join(DATA_DIR, "seed_papers.json")

# ─── Evaluation (evaluate_rag.py) ─────────────────────────────────────────────
SAMPLE_INPUTS_PATH = os.path.join(DATA_DIR, "sample_inputs")
EVAL_RESULTS_PATH  = os.path.join(DATA_DIR, "evaluation_results.csv")

# ─── Detectron2 / layout detection ──────────────────────────────────────────
# Layout detection (figure/table crops + a VLM description of each) is the
# heaviest, most install-fragile stage: detectron2 is not on PyPI and needs a
# matching torch build. With it OFF, ingestion falls back to text-only
# extraction (PyMuPDF) — which is what the Hugging Face Space runs.
LAYOUT_DETECTION  = os.environ.get("CITATION_LAYOUT_DETECTION", "1").lower() not in (
    "0", "false", "no",
)
# May be absent: when the file is not there, layoutparser downloads the
# PubLayNet weights for DETECTRON_CONFIG on first use.
DETECTRON_WEIGHTS = os.environ.get(
    "CITATION_DETECTRON_WEIGHTS", os.path.join(PROJECT_ROOT, "model_final.pth")
)
DETECTRON_CONFIG  = "lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config"
DETECTRON_LABEL_MAP = {0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}
DETECTRON_SCORE_THRESH = 0.5

# ─── Ingestion Tunables ───────────────────────────────────────────────────────
CHUNK_MIN_LENGTH         = 10       # Discard text chunks shorter than this
EMBED_BATCH_SIZE         = 1000     # ChromaDB upsert batch size
EMBED_MAX_CHARS          = 4000     # Truncate documents to this length before embedding
SEMANTIC_CHUNKER_TYPE    = "percentile"
SEMANTIC_CHUNKER_AMOUNT  = 90       # 90th percentile breakpoint

# Figure/table handling. The layout pass always crops the image and keeps its
# caption as the searchable text. Set CITATION_FIGURE_VLM=1 to also run a VLM
# description of every crop at ingest time (one model call per figure — off by
# default; crops can be described on demand instead).
FIGURE_VLM               = os.environ.get("CITATION_FIGURE_VLM", "0").lower() in (
    "1", "true", "yes",
)

# Per-document summary written to SUMMARY_COLLECTION_NAME at ingest time — the
# stage-1 "is this paper even relevant" index. One model call per paper.
SUMMARY_MODEL            = os.environ.get("CITATION_SUMMARY_MODEL", "") or None  # None → LLM_MODEL
SUMMARY_MAX_CHARS        = int(os.environ.get("CITATION_SUMMARY_MAX_CHARS", "8000"))

# ─── Agent 5 — Batch Citer ────────────────────────────────────────────────────
# Sentences per citation-need request. One request for a whole draft makes the
# entire run hostage to a single malformed reply; smaller batches confine that
# to the batch. Too small wastes calls, since each one re-sends the framing.
CITATION_CHECK_BATCH_SIZE = 20

# ─── Search Tunables ──────────────────────────────────────────────────────────
RRF_K            = 60               # Reciprocal Rank Fusion constant
DEFAULT_TOP_K    = 3                # Default number of results to return

# Two-stage retrieval: rank documents by summary similarity, keep the top
# DOC_SELECT_K, optionally have the LLM drop the off-topic ones (DOC_GATE),
# then run the detail search only over what survives.
DOC_SELECT_K     = int(os.environ.get("CITATION_DOC_SELECT_K", "6"))
DOC_GATE         = os.environ.get("CITATION_DOC_GATE", "1").lower() in ("1", "true", "yes")

# ─── Orchestrator Tunables ────────────────────────────────────────────────────
PDF_COOLDOWN_SECONDS     = 30
DRAFT_COOLDOWN_SECONDS   = 2
MANUAL_COOLDOWN_SECONDS  = 5
DEFAULT_WORKERS          = 1

# ─── Contact Address ─────────────────────────────────────────────────────────
# Unpaywall requires a contact email on every request; OpenAlex and Crossref
# use it to route you to their faster "polite" pools. Set UNPAYWALL_EMAIL in
# your environment; the placeholder below is only a fallback so the pipeline
# does not silently send someone else's address.
UNPAYWALL_EMAIL   = os.environ.get("UNPAYWALL_EMAIL", "abcdef_12345@gmail.com")

# ─── Agent 0 — Discoverer ────────────────────────────────────────────────────
# Agent 0 searches a scholarly index whose results come back ranked by
# relevance to a natural-language query and already carry an open-access PDF
# URL — so it never has to resolve a DOI or guess a download location the way
# Agent 2 does for reference chains.
#
# Providers are tried in order until one yields a result whose PDF actually
# downloads. arXiv is first: for a physics tool its preprints are almost always
# what you want and the PDF never 404s, whereas OpenAlex's top hits are often
# paywalled publisher links. OpenAlex (better metadata, no rate limit) and
# Semantic Scholar (keyless pool heavily throttled — set S2_API_KEY) back it up.
SEARCH_PROVIDERS  = os.environ.get(
    "CITATION_SEARCH_PROVIDERS", "arxiv,openalex,semanticscholar"
).split(",")
SEARCH_LIMIT      = 15    # Candidates to rank through looking for an OA PDF

OPENALEX_URL      = "https://api.openalex.org/works"
# OpenAlex gives requests that supply a contact address the faster "polite"
# pool. Reuses the same address Unpaywall needs.
OPENALEX_MAILTO   = os.environ.get("OPENALEX_MAILTO", UNPAYWALL_EMAIL)

S2_API_KEY        = os.environ.get("S2_API_KEY")
S2_SEARCH_URL     = "https://api.semanticscholar.org/graph/v1/paper/search"

# ─── Agent 1 — Extractor (GROBID) ───────────────────────────────────────────
# Default is the public GROBID instance running as a Hugging Face Space, so the
# pipeline works with no local Java service. Point this at http://localhost:8070
# when running your own GROBID (faster, private, no shared rate limit).
GROBID_SERVER            = os.environ.get("GROBID_SERVER", "https://kermitt2-grobid.hf.space")
GROBID_BATCH_CONCURRENCY = int(os.environ.get("GROBID_BATCH_CONCURRENCY", "2"))

# ─── Agent 2 — Fetcher ───────────────────────────────────────────────────────
MAX_CITATION_LEN   = 500   # Skip citations longer than this (likely malformed)
ARXIV_RATE_LIMIT   = 3     # Seconds between arXiv requests
UNPAYWALL_SLEEP    = 0.5   # Courtesy sleep after Unpaywall downloads
# ─── Rendering / DPI ─────────────────────────────────────────────────────────
PDF_RENDER_DPI = 72
