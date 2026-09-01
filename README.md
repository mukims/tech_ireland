---
title: Citation Agent
emoji: 📚
colorFrom: indigo
colorTo: pink
sdk: docker
app_port: 7860
pinned: false
---

# Citation Agent Pipeline

A multi-agent pipeline that builds a citable knowledge base from physics papers
and helps you cite it while you write. Starting from a one-line research idea,
it finds a seed paper, walks its reference list, fetches the open-access PDFs it
can find, ingests them into a hybrid (vector + keyword) search index, and then
suggests formal LaTeX citations for draft text grounded in that corpus.

It runs two ways from the same code:

- **Local** — models through [Ollama](https://ollama.com), layout analysis with
  Detectron2, your own GROBID server. Nothing leaves the machine except the
  bibliographic lookups.
- **Hosted (Hugging Face Spaces)** — a Streamlit app ([app.py](app.py)) served
  from a Docker Space ([Dockerfile](Dockerfile)), chat + embeddings through a
  hosted API, the public GROBID Space, text-only ingestion. See
  [Deploying to Hugging Face Spaces](#deploying-to-hugging-face-spaces).

The bibliographic lookups (Agent 0: OpenAlex / Semantic Scholar; Agent 2:
Crossref, Unpaywall, Europe PMC, arXiv) always go out to those services.

## Pipeline

| Stage | Script | What it does |
|-------|--------|--------------|
| **Agent 0 — Discoverer** | [agent0_discoverer.py](agent0_discoverer.py) | Takes a free-text research idea, searches a relevance-ranked scholarly index (OpenAlex, then Semantic Scholar), picks the top result that has an open-access PDF, and downloads it into `raw/` under a `source_key`-derived name. Records each query → chosen paper in `seed_papers.json`. From here the rest of the pipeline runs unchanged. |
| **Agent 1 — Extractor** | [agent1_extractor.py](agent1_extractor.py) | Sends every PDF in `raw/` to a local GROBID server, parses the TEI output, and writes each paper's own metadata plus its full reference list (title, authors, year, DOI, raw string) to `extracted_citations.json`. Scores each consolidated DOI against the printed reference so grey-literature mismatches can be flagged. |
| **Agent 2 — Fetcher** | [agent2_fetcher.py](agent2_fetcher.py) | Collapses the references to distinct sources, resolves a DOI per source (trusting Agent 1 when it was confident, otherwise asking Crossref), and tries to download an open-access PDF from Unpaywall → Europe PMC → arXiv. Writes `downloaded.json` / `failed_downloads.json` incrementally so a crashed run resumes. |
| **Agent 3 — Ingestor** | [agent3_ingestor.py](agent3_ingestor.py) | For each downloaded PDF: Detectron2 page-layout detection, a VLM pass to describe figures and tables, semantic chunking of the text, embedding into ChromaDB, and a rebuild of the BM25 index. Tracks what has already been ingested so re-runs are cheap. |
| **Agent 4 — Assistant** | [agent4_assistant.py](agent4_assistant.py) | Given a snippet of draft text, runs hybrid search over the corpus and asks the LLM to rewrite the snippet with the correct `\cite{key}` inserted, plus an explanation of why that source supports the claim. |

Agent 0 just leaves a PDF in `raw/`, which is exactly what Agent 1 already
reads — nothing downstream needs to know it ran. The reference-list format
written by Agent 1 is consumed by Agent 2, whose manifest is consumed by
Agent 3, whose index is queried by Agent 4.

**[orchestrate.py](orchestrate.py)** runs the whole chain for one research idea
as a **LangGraph** state machine:

```
discover ─(no seed)─► END
   ▼
ingest_seed ─► extract ─(GROBID down)─► END
                 ▼
               fetch ─► ingest_refs ─(--ask)─► respond ─► END
                             └───────────────────────────► END
```

Each node wraps the same agent entry point you'd otherwise call by hand; the
conditional edges handle "no seed paper found" and "GROBID produced nothing".
Every agent keeps its own on-disk state, so a run that dies partway is resumed
by running it again — finished stages no-op.

```bash
python orchestrate.py --query "topological protection in disordered quantum wires"
python orchestrate.py --query "..." --ask --workers 4
```

Configuration in [config.py](config.py) and prompts in [prompts.py](prompts.py)
also anticipate further agents (batched citation-need checking, single-file
ingestion, a research chat) and a Ragas-based evaluation script that share the
same `shared/` primitives.

## Layout

```
agent0_discoverer.py … agent4_assistant.py  the pipeline stages
orchestrate.py                              LangGraph that runs all stages for one research idea
config.py                                   all model names, paths, tunables — the one file to edit
prompts.py                                  every prompt sent to a model
shared/
  ingestion.py    PDF processing, chunking, ChromaDB upsert, BM25 rebuild
  search.py       hybrid BM25 + dense retrieval with Reciprocal Rank Fusion
  fetch.py        stream-a-PDF-to-disk-with-validation (shared by Agent 0 and Agent 2)
  db.py           ChromaDB + BM25 loading helpers
  source_key.py   deterministic identity for a reference / document (doi:, arxiv:, title:, raw:)
  log.py          console + rotating-file logging
  retry.py        exponential-backoff retry decorator
raw/                PDFs to process (Agent 0 writes here; Agent 1 reads); GROBID TEI cached in raw/grobid_output/
seed_papers.json    Agent 0's manifest: research query → chosen seed paper
pulled_pdfs/        PDFs downloaded by Agent 2
physics_vectordb/   persistent ChromaDB store + ingestion manifest
images/             figure/table crops written during ingestion (debug artefact, safe to delete)
logs/               citation_agent.log
app.py              Streamlit UI / Hugging Face Spaces entry point
model_final.pth     Detectron2 / PubLayNet checkpoint (optional; layout detection)
```

Everything the pipeline **writes** lives under `CITATION_DATA_DIR` (default: the
repo root) — set it to a writable path such as `/data` on a read-only or
ephemeral host.

## Prerequisites

- Python 3.12: `pip install -r requirements.txt`.
- **An LLM + embedding backend** (`LLM_BACKEND`, `EMBED_BACKEND`):
  - `ollama` (default) — a local [Ollama](https://ollama.com) daemon with the
    models in `config.py` pulled (`gemma4:e2b`, `qwen2.5:7b`, `nomic-embed-text`).
  - `openai` — any OpenAI-compatible endpoint (`OPENAI_BASE_URL`,
    `OPENAI_API_KEY`); `huggingface` for embeddings via `huggingface_hub`.
- **GROBID** for Agent 1 — defaults to the public `kermitt2-grobid.hf.space`;
  set `GROBID_SERVER=http://localhost:8070` to use your own
  (`curl localhost:8070/api/isalive`).
- **Layout detection is optional** (`CITATION_LAYOUT_DETECTION=1`). It needs
  `pip install -r requirements-layout.txt` plus detectron2 from source, and a
  torch build. With it off, ingestion is text-only (PyMuPDF).

## Usage

The whole pipeline for one idea:

```bash
python orchestrate.py --query "topological protection in disordered quantum wires" --ask
```

Or run the stages by hand:

```bash
# 0. Seed from a research idea — finds and downloads a relevant paper into raw/
python agent0_discoverer.py --query "topological protection in disordered quantum wires"
#    (or skip this and drop your own PDF(s) into raw/ by hand)

# 1. Mine the reference list of everything in raw/
python agent1_extractor.py            # -> extracted_citations.json

# 2. Download the open-access PDFs of those references
export UNPAYWALL_EMAIL="you@example.com"   # Unpaywall requires a contact address
python agent2_fetcher.py              # -> pulled_pdfs/, downloaded.json, failed_downloads.json

# 3. Ingest the downloaded PDFs into the search index
python agent3_ingestor.py            # -> physics_vectordb/, bm25_index.pkl
python agent3_ingestor.py --workers 4 --force   # parallel parse, re-ingest everything

# 4. Ask for a citation for a piece of draft text
python agent4_assistant.py --text "Anderson localization suppresses diffusion in 1D." --top_k 3
```

Or the Streamlit UI (same thing, with a browser front-end):

```bash
streamlit run app.py
```

## Deploying to Hugging Face Spaces

This deploys as a **Docker** Space — [Dockerfile](Dockerfile) runs the Streamlit
app. Detectron2, Ollama and a local GROBID are all avoided in this config.

```bash
# 1. commit
git add -A && git commit -m "deploy"

# 2. create a Docker Space at https://huggingface.co/new-space  (SDK: Docker),
#    then add it as a remote and push:
git remote add space https://huggingface.co/spaces/<username>/citation-agent
git push space main        # username + a write token when prompted
```

The `sdk: docker` / `app_port: 7860` front matter at the top of this file is
what configures the Space.

Then set **Settings → Variables and secrets**:

| Variable | Value | Why |
|----------|-------|-----|
| `HF_TOKEN` | *(secret)* | Auth for the router and HF-hosted embeddings |
| `OPENAI_BASE_URL` | `https://router.huggingface.co/v1` | HF Inference router (or any OpenAI-compatible URL) |
| `CITATION_LLM_MODEL` | e.g. `meta-llama/Llama-3.1-8B-Instruct` | A chat model the endpoint serves |
| `CITATION_EMBED_MODEL` | e.g. `BAAI/bge-small-en-v1.5` | An embedding model with an HF Inference endpoint |
| `UNPAYWALL_EMAIL` | your email | Required by Unpaywall / OpenAlex |
| `CITATION_DATA_DIR` | `/data` | **only if** you attach persistent storage — otherwise leave unset |
| `GROBID_SERVER` | `https://kermitt2-grobid.hf.space` | Default already; a private GROBID Space is faster and unshared |

`LLM_BACKEND=openai`, `CITATION_EMBED_BACKEND=huggingface` and
`CITATION_LAYOUT_DETECTION=0` are baked into the Dockerfile, so the table above
is the minimum you must add.

Without persistent storage the ChromaDB corpus is rebuilt each time the Space
restarts. The public GROBID Space is shared and rate-limited — for anything
beyond a demo, duplicate it into your own Space and point `GROBID_SERVER` at it.

## Configuration notes

- `config.py` is the single place for model names, directory paths, and every
  tunable (chunking thresholds, rate limits, RRF constant, batch sizes).
- Writes are anchored to `CITATION_DATA_DIR` (default: repo root); code is
  anchored to the repo, so scripts can be run from anywhere.
- Backend: `LLM_BACKEND`, `EMBED_BACKEND`, `OPENAI_BASE_URL`, `OPENAI_API_KEY` /
  `HF_TOKEN`, `CITATION_LLM_MODEL`, `CITATION_CHAT_MODEL`, `CITATION_EMBED_MODEL`.
- Ingestion: `CITATION_LAYOUT_DETECTION`, `CITATION_DETECTRON_WEIGHTS`,
  `CITATION_IMAGES_DIR`.
- Agent 1: `GROBID_SERVER`, `GROBID_BATCH_CONCURRENCY`.
- Agent 0: `CITATION_SEARCH_PROVIDERS` (default `openalex,semanticscholar`,
  tried in order), `OPENALEX_MAILTO`, `S2_API_KEY` (optional Semantic Scholar
  key — the keyless pool is heavily rate-limited).
- Misc: `UNPAYWALL_EMAIL`, `CITATION_LOG_DIR`, `CITATION_LOG_FILE=0`.
