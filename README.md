# Research Assistant Pipeline

A multi-agent pipeline that builds a citable knowledge base from physics papers.
From a one-line research idea it finds a seed paper, walks its reference list,
fetches the open-access PDFs it can find, and ingests them — chunk-level into a
hybrid (vector + keyword) index and paper-level as a one-paragraph summary. It
then answers two questions: *what has already been done on this idea* (a
related-work synthesis) and *which source backs this sentence* (a LaTeX
citation).

Retrieval is two-stage: match the idea against the paper **summaries** first,
let the LLM drop the off-topic ones, then run the detailed chunk search only
over what survives — so it stays fast as the corpus grows.

Design rationale for every major choice is in [ARCHITECTURE.md](ARCHITECTURE.md).

Models run locally through [Ollama](https://ollama.com) by default, or against
any OpenAI-compatible API (`LLM_BACKEND=openai`). A Streamlit UI
([app.py](app.py)) and a [Dockerfile](Dockerfile) are included. The
bibliographic lookups (Agent 0: arXiv / OpenAlex / Semantic Scholar; Agent 2:
Crossref, Unpaywall, Europe PMC, arXiv) always go out to those services.

## Pipeline

| Stage | Script | What it does |
|-------|--------|--------------|
| **Agent 0 — Discoverer** | [agent0_discoverer.py](agent0_discoverer.py) | Takes a free-text research idea and searches relevance-ranked indexes in turn — arXiv, then OpenAlex, then Semantic Scholar — downloading the first result whose PDF actually fetches (paywalled publisher links are skipped, not fatal). Saves it into `raw/` under a `source_key`-derived name and records the query → paper in `seed_papers.json`. If nothing downloads, `--seed-url` / `discover_from_url()` takes an arXiv or direct-PDF link instead. From here the rest of the pipeline runs unchanged. |
| **Agent 1 — Extractor** | [agent1_extractor.py](agent1_extractor.py) | Sends every PDF in `raw/` to a local GROBID server, parses the TEI output, and writes each paper's own metadata plus its full reference list (title, authors, year, DOI, raw string) to `extracted_citations.json`. Scores each consolidated DOI against the printed reference so grey-literature mismatches can be flagged. |
| **Agent 2 — Fetcher** | [agent2_fetcher.py](agent2_fetcher.py) | Collapses the references to distinct sources, resolves a DOI per source (trusting Agent 1 when it was confident, otherwise asking Crossref), and tries to download an open-access PDF from Unpaywall → Europe PMC → arXiv. Writes `downloaded.json` / `failed_downloads.json` incrementally so a crashed run resumes. |
| **Agent 3 — Ingestor** | [agent3_ingestor.py](agent3_ingestor.py) | For each downloaded PDF: semantic chunking of the text, embedding into ChromaDB, a rebuild of the BM25 index, and **one LLM summary per paper** into a separate `physics_summaries` collection. With layout detection on, it also crops figures/tables and keeps their captions (a VLM description of each is opt-in, `CITATION_FIGURE_VLM=1`). Tracks what's ingested so re-runs are cheap. |
| **Two-stage retrieval** | [shared/retrieve.py](shared/retrieve.py) | Stage 1: rank papers by summary similarity → LLM gate ("relevant prior work? Y/N") → shortlist. Stage 2: hybrid chunk search restricted to the shortlist. `research_answer()` then synthesises a related-work overview. |
| **Agent 4 — Assistant** | [agent4_assistant.py](agent4_assistant.py) | Given a snippet of draft text, runs hybrid search over the corpus and asks the LLM to rewrite the snippet with the correct `\cite{key}` inserted, plus an explanation of why that source supports the claim. |

Agent 0 just leaves a PDF in `raw/`, which is exactly what Agent 1 already
reads — nothing downstream needs to know it ran. The reference-list format
written by Agent 1 is consumed by Agent 2, whose manifest is consumed by
Agent 3, whose two indexes (chunks + summaries) are queried by the retrieval
layer and Agent 4.

**[orchestrate.py](orchestrate.py)** runs the whole chain for one research idea
as a **LangGraph** state machine:

```
discover ─(no seed, --ask)─► fallback ─► END      answer from the model alone
   │      ─(no seed)──────────────────► END
   ▼
ingest_seed ─► extract ─(GROBID down)─► END
                 ▼
               fetch ─► ingest_refs ─(--ask)─► respond ─► END
                             └───────────────────────────► END
```

Each node wraps the same agent entry point you'd otherwise call by hand; the
conditional edges handle "no seed paper found" (answer from general knowledge,
still asking for a PDF link) and "GROBID produced nothing".
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
  retrieve.py     two-stage retrieval (summary shortlist → deep chunk search)
  llm.py          backend-agnostic chat + embeddings (ollama / openai / huggingface)
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
logs/               research_assistant.log
app.py              Streamlit UI (also the container entry point)
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
#    no open-access hit? give it a link:
python agent0_discoverer.py --query "..." --url https://arxiv.org/abs/2401.12345
#    (or skip Agent 0 and drop your own PDF(s) into raw/ by hand)

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

## Configuration notes

- `config.py` is the single place for model names, directory paths, and every
  tunable (chunking thresholds, rate limits, RRF constant, batch sizes).
- Writes are anchored to `CITATION_DATA_DIR` (default: repo root); code is
  anchored to the repo, so scripts can be run from anywhere.
- Backend: `LLM_BACKEND`, `EMBED_BACKEND`, `OPENAI_BASE_URL`, `OPENAI_API_KEY` /
  `HF_TOKEN`, `CITATION_LLM_MODEL`, `CITATION_CHAT_MODEL`, `CITATION_EMBED_MODEL`.
- Ingestion: `CITATION_LAYOUT_DETECTION`, `CITATION_FIGURE_VLM`,
  `CITATION_DETECTRON_WEIGHTS`, `CITATION_IMAGES_DIR`,
  `CITATION_SUMMARY_MODEL`, `CITATION_SUMMARY_MAX_CHARS`.
- Retrieval: `CITATION_DOC_SELECT_K` (stage-1 shortlist size),
  `CITATION_DOC_GATE` (LLM relevance gate, default on).
- Agent 1: `GROBID_SERVER`, `GROBID_BATCH_CONCURRENCY`.
- Agent 0: `CITATION_SEARCH_PROVIDERS` (default `arxiv,openalex,semanticscholar`,
  tried in order), `OPENALEX_MAILTO`, `S2_API_KEY` (optional Semantic Scholar
  key — the keyless pool is heavily rate-limited).
- Misc: `UNPAYWALL_EMAIL`, `CITATION_LOG_DIR`, `CITATION_LOG_FILE=0`.
