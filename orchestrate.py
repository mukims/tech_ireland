"""
orchestrate.py — run the whole pipeline for one research idea, as a LangGraph.

    python orchestrate.py --query "topological protection in disordered wires"
    python orchestrate.py --query "..." --ask --workers 4 --force

The graph:

    START
      │
      ▼
    discover ──(no seed found)──────────────► END
      │
      ▼
    ingest_seed        index the seed paper itself, not just what it cites
      │
      ▼
    extract  ──(GROBID produced nothing)────► END
      │
      ▼
    fetch              open-access PDFs for the seed's references   (Agent 2)
      │
      ▼
    ingest_refs        layout-detect, chunk, embed everything fetched (Agent 3)
      │
      ├──(--ask)──► respond ──► END
      └─────────────────────────► END

Each node wraps the same agent entry point you'd otherwise call by hand, and
every agent keeps its own on-disk state (seed_papers.json,
extracted_citations.json, downloaded.json, the ChromaDB store) — so a run that
dies partway is resumed by simply running this again.
"""

import argparse
import hashlib
import multiprocessing
import os
from typing import Optional, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, StateGraph

import agent0_discoverer
import agent1_extractor
import agent2_fetcher
import agent3_ingestor
from config import EXTRACTED_CITATIONS_PATH, GROBID_SERVER
from shared.ingestion import ingest_pdfs
from shared.log import get_logger

logger = get_logger("orchestrate")


# ─── Graph state ─────────────────────────────────────────────────────────────


class _Inputs(TypedDict):
    query: str
    workers: int
    force: bool
    ask: bool


class PipelineState(_Inputs, total=False):
    # Optional: seed straight from this PDF / arXiv link instead of searching
    # (the fallback when the search finds nothing open-access).
    seed_url: Optional[str]
    # Filled in as the graph runs.
    seed_path: Optional[str]
    seed_label: Optional[str]
    references_ok: bool
    answer: Optional[dict]
    stopped: Optional[str]   # reason the run ended early, if it did


def _banner(title: str) -> None:
    logger.info("─" * 60)
    logger.info(title)
    logger.info("─" * 60)


# ─── Nodes ───────────────────────────────────────────────────────────────────


def discover(state: PipelineState) -> dict:
    query, force = state["query"], state.get("force", False)
    seed_url = (state.get("seed_url") or "").strip()

    if seed_url:
        _banner("discover — seeding from the supplied link")
        path = agent0_discoverer.discover_from_url(query, seed_url, force=force)
        fail = f"could not download a PDF from {seed_url}"
    else:
        _banner("discover — finding a seed paper")
        path = agent0_discoverer.discover(query, force=force)
        fail = (
            "no open-access PDF found for that query — supply an arXiv or "
            "open-access PDF link to seed from directly"
        )

    if not path:
        return {"seed_path": None, "stopped": fail}

    seed = agent0_discoverer.get_seed(query) or {}
    label = seed.get("title") or seed.get("key") or os.path.basename(path)
    return {"seed_path": path, "seed_label": label}


def ingest_seed(state: PipelineState) -> dict:
    _banner("ingest_seed — indexing the seed paper itself")
    ingest_pdfs(
        {state["seed_path"]: state["seed_label"]},
        workers=state.get("workers", 1),
        skip_ingested=not state.get("force", False),
    )
    return {}


def extract(state: PipelineState) -> dict:
    _banner("extract — mining the seed's reference list (GROBID)")
    agent1_extractor.run_extractor()
    if not os.path.exists(EXTRACTED_CITATIONS_PATH):
        return {
            "references_ok": False,
            "stopped": (
                "GROBID returned no references — the server at "
                f"{GROBID_SERVER} is down, asleep, or rate-limiting. "
                f"Check: curl {GROBID_SERVER}/api/isalive"
            ),
        }
    return {"references_ok": True}


def fetch(state: PipelineState) -> dict:
    _banner("fetch — downloading the referenced papers (Agent 2)")
    agent2_fetcher.fetch_papers()
    return {}


def ingest_refs(state: PipelineState) -> dict:
    _banner("ingest_refs — ingesting the reference PDFs (Agent 3)")
    agent3_ingestor.run_ingestor(
        workers=state.get("workers", 1), force=state.get("force", False)
    )
    return {}


def respond(state: PipelineState) -> dict:
    _banner("respond — answering the query against the new corpus (Agent 4)")
    import agent4_assistant  # imported here so corpus-building doesn't load Ollama

    result = agent4_assistant.suggest_citation(state["query"])
    return {"answer": result}


# ─── Edges ───────────────────────────────────────────────────────────────────


def _after_discover(state: PipelineState) -> str:
    return "ingest_seed" if state.get("seed_path") else END


def _after_extract(state: PipelineState) -> str:
    return "fetch" if state.get("references_ok") else END


def _after_ingest_refs(state: PipelineState) -> str:
    return "respond" if state.get("ask") else END


def build_graph():
    g = StateGraph(PipelineState)
    g.add_node("discover", discover)
    g.add_node("ingest_seed", ingest_seed)
    g.add_node("extract", extract)
    g.add_node("fetch", fetch)
    g.add_node("ingest_refs", ingest_refs)
    g.add_node("respond", respond)

    g.add_edge(START, "discover")
    g.add_conditional_edges("discover", _after_discover, ["ingest_seed", END])
    g.add_edge("ingest_seed", "extract")
    g.add_conditional_edges("extract", _after_extract, ["fetch", END])
    g.add_edge("fetch", "ingest_refs")
    g.add_conditional_edges("ingest_refs", _after_ingest_refs, ["respond", END])
    g.add_edge("respond", END)

    return g.compile(checkpointer=MemorySaver())


# ─── Entry point ─────────────────────────────────────────────────────────────


def run(
    query: str,
    workers: int = 1,
    force: bool = False,
    ask: bool = False,
    seed_url: str | None = None,
) -> int:
    app = build_graph()
    thread_id = hashlib.sha1(query.encode()).hexdigest()[:12]
    config = {"configurable": {"thread_id": thread_id}}

    final: PipelineState = {}
    for update in app.stream(
        {
            "query": query,
            "workers": workers,
            "force": force,
            "ask": ask,
            "seed_url": seed_url,
        },
        config=config,
        stream_mode="values",
    ):
        final = update

    if final.get("stopped"):
        logger.error("Pipeline stopped: %s", final["stopped"])
        return 1

    result = final.get("answer")
    if result:
        print("\n--- Grounding references ---")
        for cit in result["citations"]:
            print(f" > {cit}")
        print("\n=== SUGGESTION ===")
        print(result["suggestion"])
        print("==================\n")
    elif ask:
        logger.info("Nothing in the corpus matched the query yet.")

    logger.info("Pipeline complete for: %s", query)
    return 0


def main():
    parser = argparse.ArgumentParser(description="Run the full pipeline for one research idea.")
    parser.add_argument("--query", required=True, help="The research idea to seed from.")
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Parallel worker processes for the ingestion stages.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run every stage even if its output already exists.",
    )
    parser.add_argument(
        "--ask", action="store_true",
        help="After building the corpus, answer the original query with Agent 4.",
    )
    parser.add_argument(
        "--seed-url",
        help="Seed from this PDF / arXiv link instead of searching for a paper.",
    )
    args = parser.parse_args()

    raise SystemExit(run(
        args.query, workers=args.workers, force=args.force, ask=args.ask,
        seed_url=args.seed_url,
    ))


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
