"""
graph/tracing_config.py

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
  2. `RetryPolicy` field names / `add_node(retry_policy=...)` — CONFIRMED
     against real langgraph 1.2.9's actual signature (build_graph.py had
     the wrong kwarg name, `retry=`, fixed there).
  3. `settings.max_concurrent_llm_calls` — add this to config.py if it
     doesn't exist yet; defaults to 10 here if missing. This is THE knob
     that trades latency against rate-limit risk — tune it against your
     real Groq/Gemini/Tavily plan limits, not a guess.
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass

try:
    from corpmind.config import settings  
    import logging 

    logger = logging.getLogger(__name__)
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
#defensive wrapper (protective layer)
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
    from langsmith import traceable as _real_traceable  # type: ignore
    from langsmith.run_helpers import get_current_run_tree  # type: ignore

    _HAS_LANGSMITH = True
except ModuleNotFoundError:
    _HAS_LANGSMITH = False
#defensive wrapper (protective layer)
    def _real_traceable(*d_args, **d_kwargs):  # type: ignore
        def decorator(fn):
            return fn

        if d_args and callable(d_args[0]):
            return d_args[0]
        return decorator

    def get_current_run_tree():  # type: ignore
        return None

#defensive wrapper (protective layer)
def traceable(*d_args, **d_kwargs):
    """
    Guarded wrapper around langsmith's real @traceable. Only actually
    invokes the real decorator's wrapped call when tracing is explicitly
    enabled (LANGCHAIN_TRACING_V2=true, set by configure_tracing()) —
    otherwise calls the original function directly, bypassing langsmith
    entirely.

    WHY THIS EXISTS, CONCRETELY: with real langsmith installed but tracing
    never configured (no LANGCHAIN_API_KEY / configure_tracing() not
    called), the real @traceable decorator was observed silently returning
    None from an async node instead of the node's actual return value
    (confirmed via langgraph's compile(debug=True) step trace showing
    {'ingestion': None} for a node whose own body unconditionally returns a
    dict). Root-caused to the langsmith wrapper's interaction with an
    unconfigured/no-op tracing backend — not to graph wiring, Send
    dispatch, or state schema, all of which were verified correct in
    isolation first. This guard makes tracing a true opt-in: absent or
    off, every node behaves exactly as if @traceable were never applied.
    Re-verify this doesn't mask a DIFFERENT problem once
    LANGCHAIN_API_KEY/configure_tracing() are actually wired for real — at
    that point tracing turns on and starts exercising the real decorator
    path again, which hasn't been proven safe under real production
    tracing yet, only proven NOT silently broken when tracing is off.
    """
    import inspect

    def decorator(fn):
        traced_fn = _real_traceable(*d_args, **d_kwargs)(fn) if _HAS_LANGSMITH else fn

        if inspect.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def async_wrapper(*args, **kwargs):
                if os.environ.get("LANGCHAIN_TRACING_V2") == "true":
                    return await traced_fn(*args, **kwargs)
                return await fn(*args, **kwargs)

            return async_wrapper

        @functools.wraps(fn)
        def sync_wrapper(*args, **kwargs):
            if os.environ.get("LANGCHAIN_TRACING_V2") == "true":
                return traced_fn(*args, **kwargs)
            return fn(*args, **kwargs)

        return sync_wrapper

    if d_args and callable(d_args[0]) and not d_kwargs:
        # used as bare @traceable (no call/parens)
        return decorator(d_args[0])
    return decorator


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
# Graceful Degradation 

class TransientAPIError(Exception):
    """Class 1 — network/5xx/timeout/429. Retryable, backoff+jitter, cap 3."""


class VectorStoreFatalError(Exception):
    """Class 3a — vector-store error. NEVER retried. Its absence from
    add_node()'s retry_policy= is the fail-fast behavior — don't attach one."""


class SchemaRepairExhaustedError(Exception):
    """Class 2's reprompt loop (inline in nodes.py, cap 2) exhausted without
    valid output."""


def classify_api_exception(exc: Exception, *, retry_after: float | None = None) -> Exception:
    name = type(exc).__name__.lower()
    message = str(exc).lower()
    
    # 1. VectorStore errors check
    if "vectorstore" in name or "chroma" in name or "collection" in message:
        return VectorStoreFatalError(str(exc))
        
    # 2. Extract Groq / OpenAI dynamic response body if available to check for 'rate_limit_exceeded'
    error_code_body = ""
    if hasattr(exc, "response") and exc.response is not None:
        try:
            body = exc.response.json()
            if isinstance(body, dict) and "error" in body:
                error_code_body = str(body["error"].get("code", "")).lower()
        except Exception:
            pass

    # 3. Rate Limit / TPM ceiling match (including Groq's 413 and rate_limit_exceeded with underscore)
    is_rate_limit = (
        retry_after is not None or 
        "429" in message or 
        "413" in message or             # Groq TPM code trap
        "rate limit" in message or 
        "rate_limit" in message or       # Underscore support
        "rate_limit_exceeded" in error_code_body or
        "tpm" in message                 # Tokens per minute boundary match
    )

    if is_rate_limit:
        return TransientAPIError(f"rate limited (retry_after={retry_after}): {exc}")
        
    # 4. Other network/transient exceptions check
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