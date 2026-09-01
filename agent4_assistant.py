import argparse

from config import DEFAULT_TOP_K
from prompts import CITATION_SUGGESTION_SYSTEM, CITATION_SUGGESTION_USER
from shared.log import get_logger
from shared.db import load_search_resources
from shared.search import hybrid_search
from shared.retry import retry
from shared.llm import chat

logger = get_logger("agent4")


@retry(max_retries=3, backoff=2.0)
def _generate_suggestion(system_prompt, user_prompt):
    """Call the LLM to generate a citation suggestion (with retry)."""
    return chat([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ])


def suggest_citation(text, top_k=DEFAULT_TOP_K, search_resources=None):
    """Retrieve context for *text* and ask the LLM to cite it.

    Args:
        text:             The draft snippet to find a citation for.
        top_k:            Number of retrieved chunks to ground the suggestion on.
        search_resources: Optional (collection, bm25, texts, metadatas) tuple
                          from load_search_resources(), so a caller running
                          several queries loads the index once.

    Returns:
        dict with ``suggestion`` (str), ``citations`` (list[str]), ``passages``
        (the retrieved hybrid-search hits) and ``response`` (the ChatResult),
        or None when retrieval found nothing.
    """
    collection, bm25, texts, metadatas = search_resources or load_search_resources()

    logger.info("Searching DB for relevant context for query: '%s'", text)
    results = hybrid_search(text, collection, bm25, texts, metadatas, top_k=top_k)
    if not results:
        logger.info("No related passages found in the database.")
        return None

    context = ""
    citations_pool = []
    for i, r in enumerate(results):
        meta = r["metadata"]
        cit = meta.get("citation_source", "Unknown Citation")
        doc_name = meta.get("document", "Unknown Document")
        if cit not in citations_pool:
            citations_pool.append(cit)
        context += f"--- Source {i+1} : Document '{doc_name}' corresponding to citation {cit} ---\n"
        context += r["text"] + "\n\n"

    logger.info("Drafting citation suggestion…")
    user_prompt = CITATION_SUGGESTION_USER.format(query=text, context=context)
    result = _generate_suggestion(CITATION_SUGGESTION_SYSTEM, user_prompt)

    return {
        "suggestion": result.content,
        "citations": citations_pool,
        "passages": results,
        "response": result,
    }


def main():
    parser = argparse.ArgumentParser(description="Citation AI Assistant")
    parser.add_argument("--text", type=str, required=True, help="Draft text you want to cite.")
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K, help="Number of retrieved components.")
    args = parser.parse_args()

    try:
        result = suggest_citation(args.text, top_k=args.top_k)
    except Exception as e:
        logger.error("Error generating suggestion: %s", e)
        return

    if result is None:
        return

    print("--- Found References ---")
    for cit in result["citations"]:
        print(f" > {cit}")

    chat_result = result["response"]
    print(
        f"\n[Token Stats] Submitted: {chat_result.prompt_tokens} | "
        f"Generated: {chat_result.completion_tokens}"
    )

    print("\n=== AI ASSISTANT SUGGESTION ===")
    print(result["suggestion"])
    print("===============================\n")


if __name__ == "__main__":
    main()
