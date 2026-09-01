"""
Streamlit front-end for the Citation Agent pipeline (also the container entry
point).

Tab 1 runs the LangGraph orchestrator from a research idea. Tab 2 is the
Agent 4 citation assistant over whatever corpus has been built so far.
"""

import json
import os

import streamlit as st

import config

st.set_page_config(page_title="Citation Agent", page_icon="📚", layout="centered")
os.makedirs(config.DATA_DIR, exist_ok=True)


# ─── Data helpers ───────────────────────────────────────────────────────────


def _graph():
    # Rebuilt per run so each build starts from a clean checkpointer.
    import orchestrate

    return orchestrate.build_graph()


def _corpus_stats():
    """(chunks, papers) — cheap, tolerant of a missing/empty store."""
    chunks = 0
    try:
        import chromadb

        chunks = (
            chromadb.PersistentClient(path=config.VECTORDB_PATH)
            .get_collection(config.COLLECTION_NAME)
            .count()
        )
    except Exception:
        pass

    papers = 0
    for path in (config.SEED_PAPERS_PATH, config.DOWNLOADED_JSON_PATH):
        try:
            with open(path) as fh:
                papers += len(json.load(fh))
        except Exception:
            pass
    return chunks, papers


@st.cache_data(ttl=60, show_spinner=False)
def _grobid_ok():
    import requests

    try:
        r = requests.get(f"{config.GROBID_SERVER}/api/isalive", timeout=8)
        return r.ok and "true" in r.text.lower()
    except Exception:
        return False


def _manifest(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:
        return {}


# ─── Rendering ──────────────────────────────────────────────────────────────

STEPS = {
    "discover":    ("🔍", "Finding a seed paper"),
    "ingest_seed": ("📥", "Indexing the seed paper"),
    "extract":     ("📖", "Extracting the reference list"),
    "fetch":       ("🌐", "Fetching referenced papers"),
    "ingest_refs": ("🧩", "Ingesting reference PDFs"),
    "respond":     ("✍️", "Answering your query"),
}


def _render_suggestion(result):
    with st.container(border=True):
        st.markdown("###### Suggested text")
        st.markdown(result["suggestion"])

    cits = result.get("citations") or []
    if cits:
        st.markdown("**Grounded in**")
        for c in cits:
            st.markdown(f"- {c}")

    passages = result.get("passages") or []
    if passages:
        with st.expander(f"Retrieved context · {len(passages)} passage(s)"):
            for i, p in enumerate(passages, 1):
                m = p.get("metadata") or {}
                st.caption(
                    f"{i}. **{m.get('citation_source', '?')}** — "
                    f"{m.get('document', '?')} · p.{m.get('page', '?')}"
                )
                st.text((p.get("text") or "")[:900])
                if i < len(passages):
                    st.divider()


def _render_seed_and_downloads(query, final):
    seeds = _manifest(config.SEED_PAPERS_PATH)
    seed = seeds.get(query) or {}
    # Query-key mismatch fallback: if there's exactly one seed on record, use it.
    if not seed and len(seeds) == 1:
        seed = next(iter(seeds.values()))

    title = seed.get("title") or final.get("seed_label")
    if title or seed:
        with st.container(border=True):
            st.markdown(f"**🌱 Seed paper** — {title or seed.get('key', '—')}")
            bits = []
            if seed.get("url"):
                bits.append(f"[PDF]({seed['url']})")
            if seed.get("arxiv_id"):
                bits.append(f"arXiv:{seed['arxiv_id']}")
            if seed.get("doi"):
                bits.append(f"doi:{seed['doi']}")
            if seed.get("key"):
                bits.append(f"`{seed['key']}`")
            if bits:
                st.caption(" · ".join(bits))

    downloaded = _manifest(config.DOWNLOADED_JSON_PATH)
    failed = _manifest(config.FAILED_DOWNLOADS_PATH)
    if downloaded or failed:
        with st.expander(
            f"📄 Reference PDFs — {len(downloaded)} fetched, {len(failed)} unavailable",
            expanded=bool(downloaded) and not final.get("answer"),
        ):
            for rec in downloaded.values():
                t = rec.get("title") or rec.get("raw_reference") or rec.get("key")
                st.markdown(f"- ✅ {t}")
            for rec in failed.values():
                t = rec.get("title") or rec.get("raw_reference") or rec.get("key")
                st.markdown(
                    f"- ⚠️ {t}  \n  <sub>{rec.get('reason', '')}</sub>",
                    unsafe_allow_html=True,
                )


def _render_shortlist(selected):
    if not selected:
        return
    st.markdown(f"**Relevant prior work** — {len(selected)} paper(s)")
    for r in selected:
        with st.expander(f"{r.get('citation') or r.get('document')}  ·  score {r.get('score', '')}"):
            st.write(r.get("summary", ""))


def _render_build(final, query):
    # Show whatever the run produced — seed, downloads — even if it stopped early.
    _render_seed_and_downloads(query, final)

    if final.get("stopped"):
        st.error(final["stopped"], icon="🛑")
        if "supply an arXiv" in final["stopped"]:
            st.info(
                "Paste an arXiv or PDF link into **Seed paper URL** above and "
                "build again.",
                icon="💡",
            )
        return

    answer = final.get("answer")
    if answer:
        _render_shortlist(answer.get("selected"))
        with st.container(border=True):
            st.markdown("###### Related work")
            st.markdown(answer["suggestion"])
        if answer.get("citations"):
            st.caption("Sources: " + " · ".join(str(c) for c in answer["citations"]))
        passages = answer.get("passages") or []
        if passages:
            with st.expander(f"Retrieved context · {len(passages)} passage(s)"):
                for i, p in enumerate(passages, 1):
                    m = p.get("metadata") or {}
                    st.caption(
                        f"{i}. **{m.get('citation_source', '?')}** — "
                        f"{m.get('document', '?')} · p.{m.get('page', '?')}"
                    )
                    st.text((p.get("text") or "")[:900])
    else:
        st.info(
            "Corpus updated. Switch to **Cite a draft** to query it.", icon="✍️"
        )


# ─── Sidebar ────────────────────────────────────────────────────────────────

with st.sidebar:
    st.subheader("Corpus")
    chunks, papers = _corpus_stats()
    c1, c2 = st.columns(2)
    c1.metric("Papers", papers)
    c2.metric("Chunks", chunks)

    st.subheader("Backend")
    st.caption(f"**Chat** — `{config.LLM_MODEL}` · {config.LLM_BACKEND}")
    st.caption(f"**Embeddings** — `{config.EMBED_MODEL}` · {config.EMBED_BACKEND}")
    st.caption(
        f"**Layout** — {'on' if config.LAYOUT_DETECTION else 'text-only'}"
    )

    grobid_up = _grobid_ok()
    st.caption(("🟢" if grobid_up else "🔴") + f" **GROBID** — {config.GROBID_SERVER}")
    if not grobid_up:
        st.warning(
            "GROBID isn't responding — reference extraction (Agent 1) will fail.",
            icon="⚠️",
        )
    if config.LLM_BACKEND == "openai" and not config.OPENAI_API_KEY:
        st.error("No OPENAI_API_KEY / HF_TOKEN set.", icon="🚫")


# ─── Main ───────────────────────────────────────────────────────────────────

st.title("📚 Citation Agent")
st.caption(
    "Give it a research idea → it builds a corpus from the literature and tells "
    "you what's already been done. Or hand it a sentence and it finds the "
    "citation."
)

tab_build, tab_cite = st.tabs(["Research a topic", "Cite a draft"])


# ─── Tab 1: build a corpus ─────────────────────────────────────────────────

with tab_build:
    with st.form("build_form"):
        query = st.text_input(
            "Research idea",
            placeholder="topological protection in disordered quantum wires",
        )
        seed_url = st.text_input(
            "Seed paper URL",
            placeholder="https://arxiv.org/abs/2401.12345 — leave blank to search automatically",
            help="Used when the search finds no open-access PDF. "
                 "Accepts an arXiv link or a direct .pdf URL.",
        )
        c1, c2 = st.columns(2)
        ask = c1.toggle("Answer my query at the end", value=True)
        force = c2.toggle("Force re-run every stage", value=False)
        submitted = st.form_submit_button("Build corpus", type="primary")

    if submitted and query.strip():
        graph = _graph()
        cfg = {"configurable": {"thread_id": query.strip()[:64]}}
        inputs = {
            "query": query.strip(),
            "workers": 1,
            "force": force,
            "ask": ask,
            "seed_url": seed_url.strip() or None,
        }

        with st.status("Running the pipeline…", expanded=True) as status:
            try:
                for update in graph.stream(inputs, cfg, stream_mode="updates"):
                    for node in update:
                        icon, label = STEPS.get(node, ("•", node))
                        st.write(f"{icon} {label}")
                final = graph.get_state(cfg).values
                if final.get("stopped"):
                    status.update(label="Stopped early", state="error")
                else:
                    status.update(label="Done", state="complete")
            except Exception as e:  # noqa: BLE001
                status.update(label="Pipeline failed", state="error")
                st.exception(e)
                final = {}

        st.session_state["build_result"] = final
        st.session_state["build_query"] = query.strip()

    if st.session_state.get("build_result"):
        _render_build(
            st.session_state["build_result"],
            st.session_state.get("build_query", ""),
        )
    elif not (submitted and query.strip()):
        st.caption(
            "Agent 0 finds a paper → Agent 1 reads its references → "
            "Agent 2 fetches them → Agent 3 indexes everything → "
            "the top papers get summarised and matched to your idea."
        )


# ─── Tab 2: cite a draft ──────────────────────────────────────────────────

with tab_cite:
    if chunks == 0:
        st.info(
            "No corpus yet — build one in **Research a topic** first.", icon="📭"
        )

    with st.form("cite_form"):
        draft = st.text_area(
            "Your sentence",
            placeholder="Anderson localization suppresses diffusive transport in one dimension.",
            height=120,
        )
        top_k = st.slider("Passages to retrieve", 1, 10, config.DEFAULT_TOP_K)
        cite_submitted = st.form_submit_button(
            "Suggest a citation", type="primary", disabled=chunks == 0
        )

    if cite_submitted and draft.strip():
        from shared.db import load_search_resources
        import agent4_assistant

        try:
            resources = load_search_resources()
            with st.spinner("Retrieving and drafting…"):
                result = agent4_assistant.suggest_citation(
                    draft.strip(), top_k=top_k, search_resources=resources
                )
            st.session_state["cite_result"] = result or "empty"
        except RuntimeError:
            st.session_state["cite_result"] = "empty"
        except Exception as e:  # noqa: BLE001
            st.exception(e)
            st.session_state["cite_result"] = None

    cr = st.session_state.get("cite_result")
    if cr == "empty":
        st.info("Nothing in the corpus matched that text.", icon="🤷")
    elif isinstance(cr, dict):
        _render_suggestion(cr)
