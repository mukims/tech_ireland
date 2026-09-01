"""
Streamlit front-end for the Citation Agent pipeline — the Hugging Face Spaces
entry point (see the `app_file: app.py` line in README.md's front matter).

Tab 1 runs the LangGraph orchestrator from a research idea. Tab 2 is the
Agent 4 citation assistant over whatever corpus has been built so far.
"""

import os

import streamlit as st

import config

st.set_page_config(page_title="Citation Agent", page_icon="📚", layout="wide")

os.makedirs(config.DATA_DIR, exist_ok=True)


# ─── Sidebar: backend status ────────────────────────────────────────────────

with st.sidebar:
    st.header("Backend")
    st.write(f"**LLM:** `{config.LLM_BACKEND}` — `{config.LLM_MODEL}`")
    st.write(f"**Embeddings:** `{config.EMBED_BACKEND}` — `{config.EMBED_MODEL}`")
    st.write(f"**GROBID:** `{config.GROBID_SERVER}`")
    st.write(f"**Layout detection:** {'on' if config.LAYOUT_DETECTION else 'off (text-only)'}")
    st.write(f"**Data dir:** `{config.DATA_DIR}`")

    if config.LLM_BACKEND == "openai" and not config.OPENAI_API_KEY:
        st.error("LLM_BACKEND=openai but no OPENAI_API_KEY / HF_TOKEN is set.")

    try:
        import chromadb

        _col = chromadb.PersistentClient(path=config.VECTORDB_PATH).get_collection(
            config.COLLECTION_NAME
        )
        st.metric("Chunks in corpus", _col.count())
    except Exception:
        st.metric("Chunks in corpus", 0)


tab_build, tab_cite = st.tabs(["🔎 Research a topic", "✍️ Cite a draft"])


# ─── Tab 1: orchestrator ────────────────────────────────────────────────────

with tab_build:
    st.subheader("Seed a corpus from a research idea")
    st.caption(
        "Agent 0 finds a relevant open-access paper → its own text is indexed → "
        "Agent 1 mines its reference list → Agent 2 fetches those papers → "
        "Agent 3 ingests them."
    )

    query = st.text_input(
        "Research idea",
        placeholder="topological protection in disordered quantum wires",
    )
    seed_url = st.text_input(
        "Seed paper URL (optional)",
        placeholder="https://arxiv.org/abs/2401.12345  —  leave blank to search automatically",
        help="Used when the automatic search finds no open-access PDF. "
             "Accepts an arXiv link or a direct .pdf URL.",
    )
    col1, col2 = st.columns(2)
    ask = col1.checkbox("Answer the query at the end (Agent 4)", value=True)
    force = col2.checkbox("Force re-run every stage", value=False)

    if st.button("Build corpus", type="primary", disabled=not query.strip()):
        import orchestrate

        graph = orchestrate.build_graph()
        cfg = {"configurable": {"thread_id": query.strip()[:64] or "default"}}
        inputs = {
            "query": query.strip(),
            "workers": 1,
            "force": force,
            "ask": ask,
            "seed_url": seed_url.strip() or None,
        }

        labels = {
            "discover": "Agent 0 — seeding from your link" if seed_url.strip()
                        else "Agent 0 — finding a seed paper",
            "ingest_seed": "Indexing the seed paper",
            "extract": "Agent 1 — extracting references (GROBID)",
            "fetch": "Agent 2 — fetching referenced papers",
            "ingest_refs": "Agent 3 — ingesting reference PDFs",
            "respond": "Agent 4 — answering the query",
        }

        with st.status("Running the pipeline…", expanded=True) as status:
            try:
                for update in graph.stream(inputs, cfg, stream_mode="updates"):
                    for node in update:
                        st.write(f"✓ {labels.get(node, node)}")
                final = graph.get_state(cfg).values
                if final.get("stopped"):
                    status.update(label="Stopped early", state="error")
                    st.error(final["stopped"])
                    if "supply an arXiv" in final["stopped"]:
                        st.info(
                            "Paste a link in **Seed paper URL** above, then "
                            "click **Build corpus** again."
                        )
                else:
                    status.update(label="Pipeline complete", state="complete")
            except Exception as e:
                status.update(label="Pipeline failed", state="error")
                st.exception(e)
                final = {}

        answer = final.get("answer")
        if answer:
            st.markdown("### Suggested citation")
            st.markdown(answer["suggestion"])
            st.markdown("**Grounding references**")
            for cit in answer["citations"]:
                st.markdown(f"- {cit}")
        elif final.get("seed_path"):
            st.success("Corpus updated. Switch to **Cite a draft** to query it.")


# ─── Tab 2: Agent 4 ─────────────────────────────────────────────────────────

with tab_cite:
    st.subheader("Insert a citation into a sentence")
    draft = st.text_area(
        "Draft text",
        placeholder="Anderson localization suppresses diffusive transport in one dimension.",
        height=140,
    )
    top_k = st.slider("Passages to retrieve", 1, 10, config.DEFAULT_TOP_K)

    if st.button("Suggest a citation", type="primary", disabled=not draft.strip()):
        from shared.db import load_search_resources
        import agent4_assistant

        try:
            resources = load_search_resources()
        except RuntimeError:
            st.warning("No corpus yet — build one in the **Research a topic** tab first.")
            st.stop()

        with st.spinner("Retrieving and drafting…"):
            result = agent4_assistant.suggest_citation(
                draft.strip(), top_k=top_k, search_resources=resources
            )

        if result is None:
            st.info("Nothing in the corpus matched that text.")
        else:
            st.markdown("### Suggestion")
            st.markdown(result["suggestion"])
            st.markdown("**Found references**")
            for cit in result["citations"]:
                st.markdown(f"- {cit}")
