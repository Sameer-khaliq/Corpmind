"""

CorpMind — Tracing + retry infrastructure config (Day 15)
============================================================

Same responsibilities as before (error taxonomy, RetryPolicy configs,
LangSmith wiring) — unchanged by the async conversion, since retries and
tracing are orthogonal to sync/async. One addition: `max_concurrent_calls()`,
because going async only reduces latency if concurrency is actually capped
sensibly — unbounded parallel Sends just moves the bottleneck to rate-limit
retries, which cost more wall-clock time than they save.

WIRING / VERIFICATION YOU MUST DO:
  1. `classify_api_exception`'s checks are generic — adjust to your real
     client SDKs' actual exception types.
  2. `RetryPolicy` field names / `add_node(retry=...)` signature — verify
     against your installed langgraph version.
  3. `settings.max_concurrent_llm_calls` — add this to config.py if it
     doesn't exist yet; defaults to 10 here if missing. This is THE knob
     that trades latency against rate-limit risk — tune it against your
     real Groq/Gemini/Tavily plan limits, not a guess.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

try:
    from corpmind.config import settings  # type: ignore
    from corpmind.logging_config import get_logger  # type: ignore

    logger = get_logger(__name__)
except ModuleNotFoundError:
    import logging

    logger = logging.getLogger(__name__)

    class _StubSettings:
        langsmith_project = "corpmind-dev"
        max_concurrent_llm_calls = 10

    settings = _StubSettings()


def max_concurrent_calls() -> int:
    """The concurrency cap for per-item async node work (extraction, Phase A,
    enrichment, evaluation). Tune against your real API plan's RPM limits —
    too high just shifts time into Class-1 retry backoff instead of saving it."""
    return int(getattr(settings, "max_concurrent_llm_calls", 10))


# ---------------------------------------------------------------------------
# RetryPolicy — real langgraph import, local stand-in fallback for sandbox
# ---------------------------------------------------------------------------

try:
    from langgraph.types import RetryPolicy  # type: ignore

    _HAS_LANGGRAPH = True
except ModuleNotFoundError:
    _HAS_LANGGRAPH = False

    @dataclass
    class RetryPolicy:  # local mirror — real machine hits the try-branch
        max_attempts: int = 3
        initial_interval: float = 0.5
        backoff_factor: float = 2.0
        max_interval: float = 30.0
        jitter: bool = True
        retry_on: tuple = (Exception,)


# ---------------------------------------------------------------------------
# LangSmith tracing — real import, no-op fallback
# ---------------------------------------------------------------------------

try:
    from langsmith import traceable  # type: ignore
    from langsmith.run_helpers import get_current_run_tree  # type: ignore

    _HAS_LANGSMITH = True
except ModuleNotFoundError:
    _HAS_LANGSMITH = False

    def traceable(*d_args, **d_kwargs):  # type: ignore
        def decorator(fn):
            return fn

        if d_args and callable(d_args[0]):
            return d_args[0]
        return decorator

    def get_current_run_tree():  # type: ignore
        return None


def configure_tracing() -> None:
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_PROJECT", getattr(settings, "langsmith_project", "corpmind-dev"))
    if _HAS_LANGSMITH and not os.environ.get("LANGCHAIN_API_KEY"):
        raise RuntimeError(
            "LANGCHAIN_API_KEY not set — tracing would silently no-op instead "
            "of failing loud. Set it, or don't call configure_tracing() yet."
        )
    logger.info("tracing configured (langsmith_installed=%s)", _HAS_LANGSMITH)


def attach_trace_metadata(model_used: str, extraction_id: str, **extra: str) -> None:
    run_tree = get_current_run_tree()
    if run_tree is None:
        return
    tags = list(getattr(run_tree, "tags", None) or [])
    tags.extend([f"model:{model_used}", f"extraction_id:{extraction_id}"])
    run_tree.tags = list(dict.fromkeys(tags))
    metadata = dict(getattr(run_tree, "metadata", None) or {})
    metadata.update({"model_used": model_used, "extraction_id": extraction_id, **extra})
    run_tree.metadata = metadata


def make_trace_tags(node_name: str, *extra: str) -> list[str]:
    return [f"node:{node_name}", *extra]


# ---------------------------------------------------------------------------
# §1.4 error taxonomy — unchanged by the async conversion
# ---------------------------------------------------------------------------


class TransientAPIError(Exception):
    """Class 1 — network/5xx/timeout/429. Retryable, backoff+jitter, cap 3."""


class VectorStoreFatalError(Exception):
    """Class 3a — vector-store error. NEVER retried. Its absence from
    add_node()'s retry= is the fail-fast behavior — don't attach one."""


class SchemaRepairExhaustedError(Exception):
    """Class 2's reprompt loop (inline in nodes.py, cap 2) exhausted without
    valid output."""


def classify_api_exception(exc: Exception, *, retry_after: float | None = None) -> Exception:
    name = type(exc).__name__.lower()
    message = str(exc).lower()
    if "vectorstore" in name or "chroma" in name or "collection" in message:
        return VectorStoreFatalError(str(exc))
    if retry_after is not None or "429" in message or "rate limit" in message:
        return TransientAPIError(f"rate limited (retry_after={retry_after}): {exc}")
    if any(tok in message for tok in ("timeout", "timed out", "connection", "5xx", "502", "503", "504")):
        return TransientAPIError(str(exc))
    return exc


GROQ_RETRY_POLICY = RetryPolicy(max_attempts=3, initial_interval=0.5, backoff_factor=2.0, jitter=True, retry_on=(TransientAPIError,))
GEMINI_RETRY_POLICY = RetryPolicy(max_attempts=3, initial_interval=0.5, backoff_factor=2.0, jitter=True, retry_on=(TransientAPIError,))
TAVILY_RETRY_POLICY = RetryPolicy(max_attempts=3, initial_interval=0.5, backoff_factor=2.0, jitter=True, retry_on=(TransientAPIError,))


if __name__ == "__main__":
    assert isinstance(classify_api_exception(Exception("Connection timeout")), TransientAPIError)
    assert isinstance(classify_api_exception(Exception("429 rate limit hit"), retry_after=2.0), TransientAPIError)
    assert isinstance(classify_api_exception(Exception("ChromaDB collection not found")), VectorStoreFatalError)
    unclassified = Exception("some unrelated bug")
    assert classify_api_exception(unclassified) is unclassified
    print("[tracing_config] PASS — exception classification correctly distinct per class.")
    print("max_concurrent_calls():", max_concurrent_calls())
    print("langgraph installed:", _HAS_LANGGRAPH, "| langsmith installed:", _HAS_LANGSMITH)