"""
Agent 0 — Seed paper discoverer.

Turns a free-text research idea into the first PDF the rest of the pipeline
needs. It searches a scholarly index (OpenAlex, then Semantic Scholar — see
SEARCH_PROVIDERS), whose results come back ranked by relevance, picks the top
result that actually has an open-access PDF, downloads it into RAW_DIR under a
source_key-derived name, and records the choice in seed_papers.json.

Nothing downstream knows Agent 0 exists: it just leaves a PDF in RAW_DIR, which
is exactly what Agent 1 already reads. The returned path also lets an
orchestrator hand the seed straight to shared.ingestion.ingest_pdfs() so the
seed paper itself is indexed, not only the papers it cites.

    python agent0_discoverer.py --query "topological protection in disordered wires"
"""

import argparse
import json
import os
import re
import time

import requests

from config import (
    RAW_DIR,
    SEED_PAPERS_PATH,
    SEARCH_PROVIDERS,
    SEARCH_LIMIT,
    OPENALEX_URL,
    OPENALEX_MAILTO,
    S2_API_KEY,
    S2_SEARCH_URL,
)
from shared.fetch import HEADERS, download_pdf, filename_for
from shared.log import get_logger
from shared.source_key import source_key

logger = get_logger("agent0")


# ─── Provider searches ───────────────────────────────────────────────────────
# Each returns a list of normalised candidate dicts, most relevant first:
#   {title, doi, arxiv_id, pmid, authors: [str], year, pdf_url}
# pdf_url may be None; pick_best() skips those.


def _search_openalex(query, limit):
    res = requests.get(
        OPENALEX_URL,
        params={"search": query, "per_page": limit, "mailto": OPENALEX_MAILTO},
        headers=HEADERS,
        timeout=20,
    )
    res.raise_for_status()
    out = []
    for w in res.json().get("results") or []:
        loc = w.get("best_oa_location") or w.get("primary_location") or {}
        ids = w.get("ids") or {}
        out.append({
            "title": w.get("display_name") or w.get("title"),
            "doi": w.get("doi"),
            "arxiv_id": None,
            "pmid": (ids.get("pmid") or "").rsplit("/", 1)[-1] or None,
            "authors": [
                a["author"]["display_name"]
                for a in w.get("authorships") or []
                if a.get("author", {}).get("display_name")
            ],
            "year": w.get("publication_year"),
            "pdf_url": loc.get("pdf_url") or (w.get("open_access") or {}).get("oa_url"),
        })
    return out


def _search_semanticscholar(query, limit, retries=4):
    headers = dict(HEADERS)
    if S2_API_KEY:
        headers["x-api-key"] = S2_API_KEY

    fields = "title,year,authors,externalIds,openAccessPdf"
    for attempt in range(retries):
        res = requests.get(
            S2_SEARCH_URL,
            params={"query": query, "limit": limit, "fields": fields},
            headers=headers,
            timeout=20,
        )
        # The keyless pool 429s constantly; back off on those, but not on any
        # other HTTP error.
        if res.status_code == 429 and attempt < retries - 1:
            wait = float(res.headers.get("Retry-After", 2 ** (attempt + 1)))
            logger.warning("Semantic Scholar rate limited; retrying in %.0fs…", wait)
            time.sleep(wait)
            continue
        res.raise_for_status()
        out = []
        for p in res.json().get("data") or []:
            ext = p.get("externalIds") or {}
            out.append({
                "title": p.get("title"),
                "doi": ext.get("DOI"),
                "arxiv_id": ext.get("ArXiv"),
                "pmid": str(ext["PubMed"]) if ext.get("PubMed") else None,
                "authors": [a["name"] for a in p.get("authors") or [] if a.get("name")],
                "year": p.get("year"),
                "pdf_url": (p.get("openAccessPdf") or {}).get("url"),
            })
        return out
    return []


def _search_arxiv(query, limit):
    import xml.etree.ElementTree as ET

    res = requests.get(
        "http://export.arxiv.org/api/query",
        params={"search_query": f"all:{query}", "max_results": limit,
                "sortBy": "relevance"},
        headers=HEADERS,
        timeout=20,
    )
    res.raise_for_status()
    ns = {"a": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(res.content)

    out = []
    for e in root.findall("a:entry", ns):
        raw_id = (e.findtext("a:id", "", ns) or "").rsplit("/abs/", 1)[-1]
        arxiv_id = raw_id.split("v")[0] or None

        pdf = None
        for link in e.findall("a:link", ns):
            if link.get("title") == "pdf":
                pdf = link.get("href", "").replace("http://", "https://")
        if pdf and not pdf.endswith(".pdf"):
            pdf += ".pdf"

        out.append({
            "title": " ".join((e.findtext("a:title", "", ns) or "").split()),
            "doi": None,
            "arxiv_id": arxiv_id,
            "pmid": None,
            "authors": [a.findtext("a:name", "", ns) for a in e.findall("a:author", ns)],
            "year": (e.findtext("a:published", "", ns) or "")[:4] or None,
            "pdf_url": pdf,
        })
    return out


_PROVIDERS = {
    "openalex": _search_openalex,
    "arxiv": _search_arxiv,
    "semanticscholar": _search_semanticscholar,
}


# ─── Selection ───────────────────────────────────────────────────────────────


def find_and_fetch_seed(query, force=False, limit=SEARCH_LIMIT):
    """Walk providers × their ranked candidates until a PDF *actually downloads*.

    A candidate whose ``pdf_url`` 404s or serves an HTML paywall page (common
    for publisher links surfaced by OpenAlex) is skipped, and the search moves
    on to the next candidate and then the next provider — so a physics query
    that OpenAlex only has paywalled still gets picked up from arXiv.

    Returns ``(candidate, key, dest_path)`` or ``(None, None, None)``.
    """
    os.makedirs(RAW_DIR, exist_ok=True)
    seen = set()

    for name in SEARCH_PROVIDERS:
        pname = name.strip()
        provider = _PROVIDERS.get(pname)
        if not provider:
            logger.warning("Unknown search provider %r — skipping.", pname)
            continue
        try:
            results = provider(query, limit)
        except requests.RequestException as exc:
            logger.warning("%s search failed: %s", pname, exc)
            continue

        n_links = 0
        for cand in results:
            if not cand.get("pdf_url"):
                continue
            key = source_key(cand)
            if not key or key in seen:
                continue
            seen.add(key)
            n_links += 1

            dest = os.path.join(RAW_DIR, filename_for(key))
            if os.path.exists(dest) and not force:
                logger.info("%s: [%s] already on disk", pname, key)
                return cand, key, dest

            ok, reason = download_pdf(cand["pdf_url"], dest)
            if ok:
                logger.info("%s: [%s] %s", pname, key, os.path.basename(dest))
                return cand, key, dest
            logger.info("%s: [%s] not downloadable — %s", pname, key, reason)

        logger.info(
            "%s: %d result(s), %d with a PDF link, none downloadable.",
            pname, len(results), n_links,
        )
    return None, None, None


# ─── Manifest ────────────────────────────────────────────────────────────────


def _load_seeds():
    if not os.path.exists(SEED_PAPERS_PATH):
        return {}
    try:
        with open(SEED_PAPERS_PATH, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        logger.warning("%s is unreadable; starting fresh.", SEED_PAPERS_PATH)
        return {}


def _save_seeds(seeds):
    with open(SEED_PAPERS_PATH, "w") as f:
        json.dump(seeds, f, indent=2, ensure_ascii=False)


def get_seed(query):
    """Return Agent 0's manifest record for *query* (key, title, path, …) or None."""
    return _load_seeds().get(query)


# ─── Direct-URL fallback ─────────────────────────────────────────────────────

_ARXIV_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5}(?:v[0-9]+)?|[a-z-]+/[0-9]{7})")


def _from_arxiv_url(url):
    """(pdf_url, arxiv_id) for an arxiv.org link, else (url, None)."""
    m = _ARXIV_RE.search(url or "")
    if not m:
        return url, None
    arxiv_id = m.group(1)
    return f"https://arxiv.org/pdf/{arxiv_id}", arxiv_id


def _download_and_record(query, key, pdf_url, seeds, force, **extra):
    """Download *pdf_url* into RAW_DIR, write the seed manifest, return the path."""
    os.makedirs(RAW_DIR, exist_ok=True)
    dest = os.path.join(RAW_DIR, filename_for(key))

    if os.path.exists(dest) and not force:
        logger.info("PDF already on disk: %s", os.path.basename(dest))
    else:
        ok, reason = download_pdf(pdf_url, dest)
        if not ok:
            logger.error("Could not download %s: %s", pdf_url, reason)
            return None
        logger.info("Saved -> %s", os.path.basename(dest))

    seeds[query] = {
        "key": key,
        "url": pdf_url,
        "path": dest,
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **extra,
    }
    _save_seeds(seeds)
    return dest


def discover_from_url(query, url, force=False):
    """Seed from a PDF link the user supplied — the fallback when the search
    turns up nothing open-access. arXiv abstract links are rewritten to the PDF.

    Returns the local PDF path, or None if the download failed.
    """
    seeds = _load_seeds()
    prior = seeds.get(query)
    if prior and not force and os.path.exists(prior.get("path", "")):
        logger.info("Already seeded for this query: %s", prior["path"])
        return prior["path"]

    pdf_url, arxiv_id = _from_arxiv_url(url.strip())
    key = f"arxiv:{arxiv_id}" if arxiv_id else (source_key({"url": pdf_url}) or "url:manual")
    logger.info("Seeding from supplied link [%s]: %s", key, pdf_url)
    return _download_and_record(
        query, key, pdf_url, seeds, force, arxiv_id=arxiv_id, source="manual-url"
    )


# ─── Entry point ─────────────────────────────────────────────────────────────


def discover(query, force=False):
    """Find the top relevant open-access paper for *query* and pull it to RAW_DIR.

    Returns the local PDF path, or None when nothing usable was found.
    """
    seeds = _load_seeds()
    prior = seeds.get(query)
    if prior and not force and os.path.exists(prior.get("path", "")):
        logger.info("Already seeded for this query: %s", prior["path"])
        return prior["path"]

    logger.info("Searching for: %s", query)
    cand, key, dest = find_and_fetch_seed(query, force=force)
    if not dest:
        logger.error("No downloadable open-access PDF for query: %s", query)
        return None

    seeds[query] = {
        "key": key,
        "title": cand.get("title"),
        "url": cand["pdf_url"],
        "path": dest,
        "doi": cand.get("doi"),
        "arxiv_id": cand.get("arxiv_id"),
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": "search",
    }
    _save_seeds(seeds)
    return dest


def main():
    parser = argparse.ArgumentParser(description="Agent 0 — Seed paper discoverer")
    parser.add_argument("--query", required=True, help="Research idea to seed from.")
    parser.add_argument(
        "--url",
        help="Seed directly from this PDF / arXiv link instead of searching.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-search and re-download even if this query was already seeded.",
    )
    args = parser.parse_args()

    if args.url:
        path = discover_from_url(args.query, args.url, force=args.force)
    else:
        path = discover(args.query, force=args.force)
    if not path:
        raise SystemExit(1)
    print(path)


if __name__ == "__main__":
    main()
