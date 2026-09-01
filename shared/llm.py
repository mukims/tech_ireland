"""
Backend-agnostic LLM + embedding access.

Every model call in the pipeline goes through ``chat()`` or ``get_embeddings()``
here rather than a provider SDK directly, so a deployment switches backend by
setting environment variables (see config.py):

    LLM_BACKEND   = ollama | openai
    EMBED_BACKEND = ollama | openai | huggingface

``ollama`` is the default and needs a local daemon. ``openai`` talks to any
OpenAI-compatible endpoint (OPENAI_BASE_URL / OPENAI_API_KEY) — on Hugging Face
Spaces that is the HF router. ``huggingface`` embeddings use
``huggingface_hub.InferenceClient`` feature-extraction.

Heavy provider SDKs are imported lazily inside each backend so a checkout using
only one of them does not need the others installed.
"""

from dataclasses import dataclass

from config import (
    LLM_BACKEND,
    EMBED_BACKEND,
    LLM_MODEL,
    EMBED_MODEL,
    OPENAI_BASE_URL,
    OPENAI_API_KEY,
    HF_TOKEN,
)
from shared.log import get_logger

logger = get_logger("llm")


@dataclass
class ChatResult:
    content: str
    prompt_tokens: object = "N/A"
    completion_tokens: object = "N/A"


# ─── Chat ────────────────────────────────────────────────────────────────────


def chat(messages, model=None, images=None) -> ChatResult:
    """Run a chat completion.

    Args:
        messages: list of ``{"role", "content"}`` dicts.
        model:    model id; defaults to config.LLM_MODEL.
        images:   optional list of local image paths for a vision request
                  (attached to the last user message).
    """
    model = model or LLM_MODEL
    if LLM_BACKEND == "openai":
        return _openai_chat(messages, model, images)
    if LLM_BACKEND == "ollama":
        return _ollama_chat(messages, model, images)
    raise ValueError(f"Unknown LLM_BACKEND {LLM_BACKEND!r} (expected 'ollama' or 'openai')")


def _ollama_chat(messages, model, images) -> ChatResult:
    import ollama

    if images:
        messages = list(messages)
        messages[-1] = {**messages[-1], "images": list(images)}
    resp = ollama.chat(model=model, messages=messages, stream=False)
    try:
        content = resp.message.content
    except AttributeError:
        content = resp["message"]["content"]
    return ChatResult(
        content=content,
        prompt_tokens=getattr(resp, "prompt_eval_count", "N/A"),
        completion_tokens=getattr(resp, "eval_count", "N/A"),
    )


def _b64_data_url(path):
    import base64
    import mimetypes

    mime = mimetypes.guess_type(path)[0] or "image/png"
    with open(path, "rb") as fh:
        return f"data:{mime};base64,{base64.b64encode(fh.read()).decode()}"


def _openai_chat(messages, model, images) -> ChatResult:
    from openai import OpenAI

    client = OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)

    if images:
        messages = list(messages)
        last = messages[-1]
        parts = [{"type": "text", "text": last["content"]}]
        parts += [
            {"type": "image_url", "image_url": {"url": _b64_data_url(p)}} for p in images
        ]
        messages[-1] = {**last, "content": parts}

    resp = client.chat.completions.create(model=model, messages=messages, stream=False)
    usage = getattr(resp, "usage", None)
    return ChatResult(
        content=resp.choices[0].message.content,
        prompt_tokens=getattr(usage, "prompt_tokens", "N/A"),
        completion_tokens=getattr(usage, "completion_tokens", "N/A"),
    )


# ─── Embeddings ──────────────────────────────────────────────────────────────
# Returns an object implementing the LangChain Embeddings interface
# (embed_documents / embed_query) so it can be handed straight to
# SemanticChunker as well as used directly.


class _InferenceClientEmbeddings:
    """huggingface_hub.InferenceClient feature-extraction as a LangChain Embeddings."""

    def __init__(self, model):
        from huggingface_hub import InferenceClient

        self._model = model
        self._client = InferenceClient(token=HF_TOKEN)

    def embed_documents(self, texts):
        out = self._client.feature_extraction(list(texts), model=self._model)
        return [row.tolist() if hasattr(row, "tolist") else list(row) for row in out]

    def embed_query(self, text):
        return self.embed_documents([text])[0]


class _OpenAIEmbeddings:
    """OpenAI-compatible /embeddings endpoint as a LangChain Embeddings."""

    def __init__(self, model):
        from openai import OpenAI

        self._model = model
        self._client = OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)

    def embed_documents(self, texts):
        resp = self._client.embeddings.create(model=self._model, input=list(texts))
        return [d.embedding for d in resp.data]

    def embed_query(self, text):
        return self.embed_documents([text])[0]


_embeddings_singleton = None


def get_embeddings(model=None):
    """Return a process-wide singleton embeddings object for EMBED_BACKEND."""
    global _embeddings_singleton
    if _embeddings_singleton is None:
        model = model or EMBED_MODEL
        if EMBED_BACKEND == "huggingface":
            logger.info("Embeddings: huggingface InferenceClient (%s)", model)
            _embeddings_singleton = _InferenceClientEmbeddings(model)
        elif EMBED_BACKEND == "openai":
            logger.info("Embeddings: OpenAI-compatible endpoint (%s)", model)
            _embeddings_singleton = _OpenAIEmbeddings(model)
        elif EMBED_BACKEND == "ollama":
            from langchain_ollama import OllamaEmbeddings

            logger.info("Embeddings: local Ollama (%s)", model)
            _embeddings_singleton = OllamaEmbeddings(model=model)
        else:
            raise ValueError(
                f"Unknown EMBED_BACKEND {EMBED_BACKEND!r} "
                "(expected 'ollama', 'openai' or 'huggingface')"
            )
    return _embeddings_singleton
