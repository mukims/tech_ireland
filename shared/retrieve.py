"""
Two-stage retrieval.

Stage 1 — cheap: rank documents by the similarity of their *summary* to the
query, keep the top DOC_SELECT_K, and (if DOC_GATE) let the LLM drop the ones
that aren't plausibly relevant prior work.

Stage 2 — expensive: hybrid chunk search, but only over the documents that
survived stage 1.

As the corpus grows this keeps the detailed search bounded to a handful of
papers instead of the whole store.
"""

from config import (
    VECTORDB_PATH,
    SUMMARY_COLLECTION_NAME,
    DOC_SELECT_K,
    DOC_GATE,
    DEFAULT_TOP_K,
)
from prompts import DOC_RELEVANCE_GATE, RESEARCH_CHAT_SYSTEM, RELATED_WORK_USER
from shared.log import get_logger

logger = get_logger("retrieve")


# ─── Stage 1 ────────────────────────────────────────────────────────────────


def rank_documents(query: str, k: int = DOC_SELECT_K) -> list[dict]:
    """Nearest document summaries. Returns [{document, citation, summary, score}]."""
    import chromadb

    from shared.llm import get_embeddings

    client = chromadb.PersistentClient(path=VECTORDB_PATH)
    try:
        col = client.get_collection(SUMMARY_COLLECTION_NAME)
    except Exception:
        logger.warning("No summary collection yet — has anything been ingested?")
        return []

    n = col.count()
    if not n:
        return []

    res = col.query(
        query_embeddings=[get_embeddings().embed_query(query)],
        n_results=min(k, n),
        include=["documents", "metadatas", "distances"],
    )
    out = []
    for summary, meta, dist in zip(
        res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        out.append({
            "document": meta.get("document", "?"),
            "citation": meta.get("citation_source", ""),
            "summary": summary,
            "score": round(1.0 - dist, 4),
        })
    return out


def gate_documents(query: str, ranked: list[dict]) -> list[dict]:
    """Ask the LLM to keep only the summaries that could be relevant prior work."""
    from shared.llm import chat

    kept = []
    for r in ranked:
        try:
            ans = chat([{
                "role": "user",
                "content": DOC_RELEVANCE_GATE.format(query=query, summary=r["summary"]),
            }]).content.strip().upper()
        except Exception as e:
            logger.warning("Gate call failed for %s: %s — keeping it.", r["document"], e)
            kept.append(r)
            continue
        if ans.startswith("Y"):
            kept.append(r)

    if not kept:
        logger.info("Gate rejected everything — falling back to the top summary.")
        return ranked[:1]
    logger.info("Gate kept %d/%d documents.", len(kept), len(ranked))
    return kept


# ─── Stage 2 ────────────────────────────────────────────────────────────────


def deep_search(query: str, documents, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """Hybrid chunk search restricted to `documents`."""
    from shared.db import load_search_resources
    from shared.search import hybrid_search

    collection, bm25, texts, metadatas = load_search_resources()
    return hybrid_search(
        query, collection, bm25, texts, metadatas,
        top_k=top_k, doc_filter=set(documents),
    )


# ─── End to end ─────────────────────────────────────────────────────────────


def research_answer(query: str, top_k: int = DEFAULT_TOP_K) -> dict | None:
    """Full pipeline: shortlist papers → deep search → related-work synthesis.

    Returns a dict shaped like agent4_assistant.suggest_citation's output
    (``suggestion`` / ``citations`` / ``passages``) plus ``selected`` — the
    stage-1 shortlist with summaries — or None when the corpus is empty.
    """
    from shared.llm import chat

    ranked = rank_documents(query)
    if not ranked:
        return None

    selected = gate_documents(query, ranked) if DOC_GATE else ranked
    docs = [r["document"] for r in selected]

    try:
        passages = deep_search(query, docs, top_k=max(top_k, len(docs)))
    except Exception as e:
        logger.warning("Deep search unavailable (%s) — using summaries only.", e)
        passages = []

    context = ""
    cites = []
    for i, p in enumerate(passages, 1):
        m = p.get("metadata") or {}
        cit = m.get("citation_source", "Unknown")
        if cit not in cites:
            cites.append(cit)
        context += f"[{cit}] (from {m.get('document', '?')}, p.{m.get('page', '?')})\n"
        context += (p.get("text") or "") + "\n\n"

    # Fall back to the summaries themselves if stage 2 found no chunks.
    if not context:
        for r in selected:
            cites.append(r["citation"])
            context += f"[{r['citation']}]\n{r['summary']}\n\n"

    answer = chat([
        {"role": "system", "content": RESEARCH_CHAT_SYSTEM},
        {"role": "user", "content": RELATED_WORK_USER.format(query=query, context=context)},
    ]).content

    return {
        "suggestion": answer,
        "citations": cites,
        "passages": passages,
        "selected": selected,
    }
