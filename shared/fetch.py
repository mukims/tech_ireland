"""
Shared PDF download helper.

The stream-to-disk-with-validation logic that Agent 0 (seed discovery) and
Agent 2 (reference fetching) both need. Kept in one place so the "the server
returned an HTML error page with HTTP 200" guard behaves identically for both
callers — a PDF that silently isn't a PDF is the single most common failure in
this pipeline.
"""

import os
import re

import requests

from config import UNPAYWALL_EMAIL
from shared.log import get_logger

logger = get_logger("fetch")

# Crossref, arXiv, Unpaywall and Semantic Scholar all ask for identifying
# contact details. Being polite gets you the faster rate-limit pool rather
# than the shared one.
HEADERS = {"User-Agent": f"citation_builder/0.1 (mailto:{UNPAYWALL_EMAIL})"}


def filename_for(key: str) -> str:
    """Stable, filesystem-safe filename derived from a source key."""
    return re.sub(r"[^a-zA-Z0-9._-]", "_", key)[:120] + ".pdf"


def _is_pdf(first_bytes: bytes) -> bool:
    """Servers frequently return an HTML error page with status 200."""
    return first_bytes[:5] == b"%PDF-"


def download_pdf(url: str, dest_path: str) -> tuple[bool, str | None]:
    """Stream a PDF to *dest_path*, refusing anything that isn't actually a PDF.

    Returns ``(ok, reason)``. On success ``reason`` is ``None``; on failure
    ``dest_path`` is left untouched (the download goes to a ``.part`` file that
    is only renamed into place once the whole body is written).

    Network errors are caught and returned as a failure reason rather than
    raised, so a caller trying several sources in turn moves on to the next one
    instead of aborting the whole fetch.
    """
    try:
        with requests.get(url, headers=HEADERS, stream=True, timeout=30) as res:
            if res.status_code != 200:
                return False, f"HTTP {res.status_code}"

            chunks = res.iter_content(chunk_size=8192)
            try:
                first = next(chunks)
            except StopIteration:
                return False, "empty response"

            if not _is_pdf(first):
                ctype = res.headers.get("Content-Type", "unknown")
                return False, f"not a PDF (content-type {ctype})"

            tmp_path = dest_path + ".part"
            with open(tmp_path, "wb") as fh:
                fh.write(first)
                for chunk in chunks:
                    fh.write(chunk)
    except requests.RequestException as exc:
        return False, f"download error: {exc}"

    os.replace(tmp_path, dest_path)
    return True, None
