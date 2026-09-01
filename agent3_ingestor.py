import os
import json
import argparse
import multiprocessing

from config import DOWNLOADED_JSON_PATH
from shared.log import get_logger
from shared.ingestion import ingest_pdfs

logger = get_logger("agent3")


def _pdfs_from_manifest(downloaded):
    """Turn agent 2's manifest into the {pdf_path: citation_label} ingest wants.

    Agent 2 keys the manifest by source_key and stores a record per paper, so
    the keys are identifiers like "doi:10.1021/ct400617g" rather than paths.
    Passing that straight through made every entry look like a missing file and
    ingested nothing at all.

    The older flat {path: citation} shape is still accepted, matching the
    tolerance agent 2 has for legacy reference files.
    """
    pdfs = {}
    for key, value in downloaded.items():
        if not isinstance(value, dict):
            pdfs[key] = value                      # legacy {path: citation}
            continue

        path = value.get("path")
        if not path:
            # Recorded as downloaded but with nowhere to read it from.
            logger.warning("No path recorded for %s — skipping.", key)
            continue

        # The citation label becomes citation_source on every chunk, and from
        # there the key in the cited draft's mapping. Prefer the parsed title
        # over the raw reference string, and never fall back to the opaque key.
        pdfs[path] = value.get("title") or value.get("raw_reference") or key

    return pdfs


def run_ingestor(workers=1, force=False):
    """Ingest every PDF Agent 2 successfully downloaded.

    Args:
        workers: Parallel worker processes for the parsing stage.
        force:   Re-process PDFs even if they are already recorded as ingested.
    """
    if not os.path.exists(DOWNLOADED_JSON_PATH):
        logger.info(
            "No download manifest at %s — run Agent 2 (python agent2_fetcher.py) "
            "first to fetch papers.", DOWNLOADED_JSON_PATH,
        )
        return

    with open(DOWNLOADED_JSON_PATH, "r") as f:
        downloaded = json.load(f)

    if not downloaded:
        logger.info("Download manifest is empty — nothing to ingest.")
        return

    if force:
        logger.info("--force: re-processing all PDFs regardless of ingestion status.")

    pdfs = _pdfs_from_manifest(downloaded)
    if not pdfs:
        logger.info("No usable PDF paths in the download manifest — nothing to ingest.")
        return

    result = ingest_pdfs(pdfs, workers=workers, skip_ingested=not force)

    logger.info(
        "Done. Processed %d, skipped %d, inserted %d chunk(s), %d unreadable.",
        result["processed"], result["skipped"], result["inserted"], len(result["failed"]),
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agent 3 — Ingestor with parallelization")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers. Use 1 to disable parallelization, 2+ for multiprocessing.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-processing of all PDFs, even if already ingested.",
    )
    args = parser.parse_args()

    multiprocessing.set_start_method("spawn", force=True)
    run_ingestor(workers=args.workers, force=args.force)
