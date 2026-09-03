# How to use the Research Assistant

This is the guide to *driving* the app. For installing and running it from a
checkout see [GETTING_STARTED.md](GETTING_STARTED.md); for what each stage does
and why, [README.md](README.md) and [ARCHITECTURE.md](ARCHITECTURE.md).

The app renders this file itself, under the **How to use** tab.

---

## What it does

You give it a one-line research idea. It finds a paper on that topic, reads that
paper's reference list, downloads the references it can legally get, and builds
a searchable corpus out of them.

Then it answers two questions:

- **What has already been done on this idea?** — a related-work synthesis.
- **What should I cite for this sentence?** — paste a sentence of draft text and
  it returns the same sentence with a `\cite{key}` inserted, plus why that
  source supports the claim.

It builds its corpus from one seed paper's references. It is not a literature
search over everything ever published — the corpus is only as broad as the seed
paper's bibliography.

---

## Before you start: read the sidebar

The sidebar is the app's status panel. Check it first, because two of the three
things there decide whether a run can succeed at all.

| Row | What it means |
|---|---|
| **Papers / Chunks** | How much is in the corpus right now. `0 / 0` means nothing has been ingested yet — Tab 2 will have nothing to search. |
| **Chat / Embeddings** | Which models are wired up, and via which backend. |
| **Layout** | `text-only` is the fast path, and what the packaged and Docker builds use. `on` means layout detection is enabled — it needs detectron2 and costs ~15s per figure. Set `CITATION_LAYOUT_DETECTION=0` unless you need figure crops. |
| **GROBID** | Green: reference extraction works. **Red: it will fail.** |

**If GROBID is red, stop.** By default the app uses a public shared GROBID
instance that throttles and goes down. A run started with GROBID red will find a
seed paper and then produce nothing from it. Either wait, or run your own:

```bash
docker run -p 8070:8070 lfoppiano/grobid:0.8.0
```

then set `GROBID_SERVER=http://localhost:8070` and restart the app.

---

## Tab 1 — Research a topic

This is the tab that builds the corpus. Everything else depends on it.

1. **Research idea** — one line of plain English. This gets used as a search
   query against arXiv, OpenAlex and Semantic Scholar, so it should read like a
   topic, not a question.

   - Good: `topological protection in disordered quantum wires`
   - Good: `spin-orbit coupling in monolayer transition metal dichalcogenides`
   - Poor: `what is the best way to protect qubits?` — conversational phrasing
     retrieves badly.

2. **Seed paper URL** *(optional)* — an arXiv link or a direct `.pdf` URL. Use
   this when you want to control which paper the corpus is built from instead of
   letting the search pick. It is also the fallback when the search finds
   nothing open-access.

3. **Answer my query at the end** *(on by default)* — runs the related-work
   synthesis once ingestion finishes. Turn it off if you only want the corpus.

4. **Force re-run every stage** *(off by default)* — ignores saved state and
   redoes everything. Normally leave this off: every stage records what it has
   already done, so a re-run resumes rather than repeats.

5. **Build corpus.** Progress appears live as each stage finishes.

### What you should expect to see

The seed paper appears first, then downloaded PDFs accumulate, then the
shortlist, then the synthesis. A normal run ends with substantially fewer papers
than the seed's reference list — most references are paywalled, and those are
skipped rather than treated as errors.

**Timing.** Minutes, not seconds. On local Ollama over CPU, expect considerably
longer — model calls dominate. The reference-fetching stage is deliberately
rate-limited to stay polite to arXiv and Unpaywall.

---

## Tab 2 — Cite a draft

**This needs a corpus.** If Papers is `0`, build one in Tab 1 first.

1. Paste a sentence of your draft — one claim, not a paragraph. The retrieval
   matches a specific assertion against specific passages, so
   `Anderson localization suppresses diffusion in 1D.` works far better than
   three sentences of background.
2. Set how many passages to retrieve. More passages give the model more to weigh
   but dilute precision; the default is a sensible starting point.
3. **Suggest a citation.**

You get the sentence back with a `\cite{key}` inserted, the sources it drew on,
and an explanation of why each supports the claim. Expand the retrieved-context
panel to see the actual passages — **do this before trusting the citation.** The
model is choosing among what retrieval handed it; if the corpus has nothing
genuinely on-point, it will still pick the closest thing.

---

## Where your work is stored

The corpus, the downloaded PDFs and the JSON manifests persist between runs.

| How you run it | Where it goes |
|---|---|
| From a checkout | The repo directory |
| Docker | Whatever you mounted at `/home/user/data` — **mount something, or it is lost** |
| Cloud Run | In memory, **lost when the instance scales to zero** |

`CITATION_DATA_DIR` overrides this everywhere.

---

## Things that go wrong

**"No open-access PDF found for that query."** The search found candidates but
every one was paywalled. Paste an arXiv or direct-PDF link into *Seed paper URL*
and run again. The app will still answer from the model's own knowledge, but
that answer is ungrounded — it is not backed by any corpus.

**GROBID is red / the run stops after the seed paper.** The shared GROBID
instance is down or throttling. See the sidebar section above.

**Most downloads fail.** Expected outside physics. arXiv coverage is excellent;
biomedical and chemistry much less so. Failures are recorded, not fatal.

**Citations look plausible but wrong.** Check the retrieved passages. A small or
off-topic corpus produces confident, badly-grounded suggestions — this is the
single most important thing to verify manually.

**It is very slow.** Local Ollama on CPU is the usual cause; a hosted backend is
dramatically faster. If Layout is not `text-only`, figure processing is costing
you roughly 15 seconds per figure.

---

## A first run, end to end

1. Check the sidebar — GROBID green, models listed.
2. Tab 1: enter `topological protection in disordered quantum wires`, leave the
   URL blank, leave both toggles as they are, **Build corpus**.
3. Wait. Watch the papers count in the sidebar climb.
4. Read the related-work synthesis at the end.
5. Tab 2: paste `Anderson localization suppresses diffusion in 1D.` and
   **Suggest a citation**.
6. Expand the retrieved passages and check the citation is actually supported.
