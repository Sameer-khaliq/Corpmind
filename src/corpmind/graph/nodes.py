"""
graph/nodes.py

CorpMind — Graph nodes (Day 14), async (latency decision, locked in)
=======================================================================

ASYNC DECISION, LOCKED IN: latency is a hard priority. Every per-item node
is `async def`. Send-fan-out only buys real wall-clock concurrency if the
nodes it dispatches to are async — sync nodes under Send still execute
serially in practice. This is why the conversion happened now (Day 14)
instead of being retrofitted at Day 20+ after every agent file already
assumed sync.

ROUTING DECISION, LOCKED IN (flips the earlier default — confirm this
replaced the old AMBIGUOUS-only-skip logic everywhere, not just here):
  - MATCHED_EXISTING -> Enrichment -> Evaluation (enrich_and_evaluate_node)
  - NEW_PRODUCT       -> Evaluation only, evaluates extraction only (evaluate_only_node)
  - AMBIGUOUS         -> Evaluation only, runs Day 13's LLM disambiguation (evaluate_only_node)
Deliberately keeps NEW_PRODUCT and AMBIGUOUS OUT of Enrichment — there's no
existing catalog entry to ground new-product enrichment against, and running
AMBIGUOUS through Enrichment would build the complex cycle this decision
exists to avoid. evaluate_only_node is intentionally the SAME node for both
— evaluate_match() already only fires disambiguation when decision ==
AMBIGUOUS, so NEW_PRODUCT passing through it is a no-op extra branch, not a
new code path.

REAL CONSTRAINT BEING FLAGGED, NOT GLOSSED OVER: your actual
evaluate_item()/evaluate_match()/default_disambiguation_fn (Days 12-13,
confirmed working on your machine) are SYNCHRONOUS — they use the sync Groq
client. Calling them directly inside an `async def` node would block the
event loop during every API call, silently defeating the entire point of
this conversion. Fix applied here: `_call_maybe_async()` runs sync callables
via `asyncio.to_thread()`, so they don't block the loop, while still
supporting real async callables directly (no thread overhead) if you later
swap in async Groq/Gemini/Tavily clients for extraction/Phase A/enrichment.
to_thread concurrency is capped by Python's default thread pool
(~min(32, cpu_count+4)) unless you configure a custom executor — good
enough to unblock the loop, NOT the same as true async I/O concurrency.
Converting evaluate_item's internals to native async (AsyncGroq client) is
the further optimization if 60-90s/100-items isn't met after this — that
touches agents/evaluation.py, which stays alone until you ask for it.

Concurrency itself is capped via a module-level asyncio.Semaphore sized from
tracing_config.max_concurrent_calls() — uncapped Send fan-out just shifts
time into Class-1 retry backoff instead of saving it.

WIRING YOU MUST DO before this runs for real:
  1. Every `_default_*_fn` below is a stub (with a small artificial
     asyncio.sleep to simulate API latency for the timing smoke test). Wire
     real agents.* calls via each `make_*_node()` factory's `*_fn=` params.
  2. `ItemState`/`BatchState` mirror Day 2's documented state.py — confirm
     your real schema matches, then delete the except-branch fallback.
"""

from __future__ import annotations

import asyncio
import inspect
import operator
import random
import time
from typing import Annotated, Any, Callable, TypedDict

from pydantic import ValidationError

from graph.tracing_config import (
    SchemaRepairExhaustedError,
    TransientAPIError,
    VectorStoreFatalError,
    attach_trace_metadata,
    classify_api_exception,
    make_trace_tags,
    max_concurrent_calls,
    traceable,
)

try:
    from corpmind.config import settings  # type: ignore
    from corpmind.logging_config import get_logger  # type: ignore

    logger = get_logger(__name__)
except ModuleNotFoundError:
    import logging

    logger = logging.getLogger(__name__)

    class _StubSettings:
        pass

    settings = _StubSettings()

try:
    from corpmind.agents.evaluation import (  # type: ignore
        EnrichmentResolution,
        EnrichmentResult,
        EnrichmentSource,
        FieldEnrichment,
        MatchDecision,
        MatchResult,
        evaluate_item,
    )
except ModuleNotFoundError:
    from evaluation_agent import (  # type: ignore  (sandbox fallback filename)
        EnrichmentResolution,
        EnrichmentResult,
        EnrichmentSource,
        FieldEnrichment,
        MatchDecision,
        MatchResult,
        evaluate_item,
    )

try:
    from corpmind.schemas.state import ItemState, BatchState  # type: ignore
except ModuleNotFoundError:

    class ItemState(TypedDict, total=False):
        catalog_id: str
        supplier_id: str
        source_row_index: int
        extraction_id: str
        raw_row: dict
        normalized_product: dict | None
        candidates: list[dict]
        match_result: dict | None
        enrichment_result: dict | None
        evaluation_record: dict | None

    class BatchState(TypedDict, total=False):
        raw_items: list[ItemState]  # set once by ingestion_node — plain overwrite
        phase_a_out: Annotated[list[ItemState], operator.add]  # Send fan-out #1 accumulator
        matched_items: list[ItemState]  # set once by phase_b_matching_node — plain overwrite, NOT accumulated
        eval_out: Annotated[list[ItemState], operator.add]  # Send fan-out #2 accumulator
        accepted: list[dict]
        review_queue: list[dict]
        report: dict | None



_SEMAPHORE = asyncio.Semaphore(max_concurrent_calls())


async def _call_maybe_async(fn: Callable, *args, **kwargs):
    """Awaits fn directly if it's already async; otherwise runs it in a
    thread so a sync/blocking call (e.g. today's real evaluate_item) doesn't
    stall the event loop for every other concurrent item."""
    if inspect.iscoroutinefunction(fn):
        return await fn(*args, **kwargs)
    return await asyncio.to_thread(fn, *args, **kwargs)




IngestionFn = Callable[..., Any]
ExtractionFn = Callable[..., Any]
PhaseAFn = Callable[..., Any]
PhaseBFn = Callable[..., Any]
EnrichmentFn = Callable[..., Any]
ReportFn = Callable[..., Any]

_STUB_LATENCY_SECONDS = 0.05  # simulates real API latency so the timing smoke test can prove concurrency actually helps


async def _default_ingestion_fn(feed_descriptor: dict) -> list[dict]:
    return feed_descriptor.get("rows", [])


async def _default_extraction_fn(raw_row: dict) -> dict:
    # WIRING: replace with your real (ideally async) Groq extraction call
    await asyncio.sleep(_STUB_LATENCY_SECONDS + random.uniform(0, 0.02))
    return {**raw_row, "field_provenance": {}, "extraction_warnings": []}


async def _default_phase_a_fn(normalized: dict) -> list[dict]:
    # WIRING: replace with your real (ideally async) vector-store lookup
    await asyncio.sleep(_STUB_LATENCY_SECONDS)
    return []


async def _default_phase_b_fn(items: list[ItemState]) -> list[ItemState]:
    # WIRING: replace with agents.matching.phase_b_cluster_and_assign(items).
    # Deliberately sequential — Phase B is single-writer by design, not a
    # concurrency target. Stub assigns MATCHED_EXISTING to every 3rd item,
    # AMBIGUOUS to every 5th, NEW_PRODUCT otherwise, so the routing decision
    # is exercised by the smoke test without needing real RRF/clustering.
    out = []
    for i, item in enumerate(items):
        item = dict(item)
        item["catalog_id"] = item.get("catalog_id") or f"cat-{i:04d}"
        if i % 5 == 4:
            decision = "AMBIGUOUS"
            catalog_id = None  # real schema forbids catalog_id when AMBIGUOUS
        elif i % 3 == 2:
            decision = "MATCHED_EXISTING"
            catalog_id = item["catalog_id"]
        else:
            decision = "NEW_PRODUCT"
            catalog_id = item["catalog_id"]
        item["match_result"] = {"catalog_id": catalog_id, "rrf_score": 0.9, "decision": decision}
        out.append(item)
    return out


async def _default_enrichment_fn(normalized: dict, candidates: list[dict]) -> dict:
    # WIRING: replace with your real (ideally async) Tavily+LLM enrichment call
    await asyncio.sleep(_STUB_LATENCY_SECONDS * 2)
    return {"catalog_id": normalized.get("catalog_id", ""), "field_results": []}


async def _default_report_fn(accepted: list[dict], review: list[dict]) -> dict:
    return {"accepted_count": len(accepted), "review_count": len(review)}


def _default_disambiguation_fn(match_result) -> dict:
    # WIRING: replace with your real llama-3.3-70b-versatile disambiguation
    # call. MUST STAY SYNC — evaluate_item (Days 12-13, real code) calls
    # this directly and synchronously, it does not await it. The outer node
    # still gets async concurrency via _call_maybe_async's asyncio.to_thread
    # wrapping of evaluate_item itself; if you want a real async Groq client
    # here, bridge it with asyncio.run(...) inside this function — safe to
    # do because this already executes inside a to_thread worker thread, not
    # the main event loop.
    time.sleep(_STUB_LATENCY_SECONDS)
    return {"resolved": True, "confidence": 0.9, "reasoning": "stub disambiguation for smoke testing"}


# ---------------------------------------------------------------------------
# Batch-level nodes
# ---------------------------------------------------------------------------


def make_ingestion_node(ingestion_fn: IngestionFn = _default_ingestion_fn) -> Callable:
    @traceable(name="ingestion_node", tags=make_trace_tags("ingestion"))
    async def _ingestion_node(state: BatchState) -> dict:
        feed_descriptor = state.get("feed_descriptor", {})  # type: ignore[typeddict-item]
        try:
            raw_rows = await _call_maybe_async(ingestion_fn, feed_descriptor)
        except Exception as e:
            raise classify_api_exception(e) from e
        items: list[ItemState] = [
            {"raw_row": row, "extraction_id": row.get("extraction_id", f"row-{i}"), "source_row_index": i}
            for i, row in enumerate(raw_rows)
        ]
        return {"raw_items": items}

    return _ingestion_node


def make_phase_b_node(phase_b_fn: PhaseBFn = _default_phase_b_fn) -> Callable:
    @traceable(name="phase_b_matching_node", tags=make_trace_tags("phase_b_matching"))
    async def _phase_b_node(state: BatchState) -> dict:
        matched = await _call_maybe_async(phase_b_fn, state.get("phase_a_out", []))
        return {"matched_items": matched}

    return _phase_b_node


def make_split_results_node() -> Callable:
    @traceable(name="split_results_node", tags=make_trace_tags("split_results"))
    async def _split_results_node(state: BatchState) -> dict:
        accepted, review = [], []
        for item in state.get("eval_out", []):
            record = item.get("evaluation_record") or {}
            (accepted if record.get("overall_verdict") == "ACCEPT" else review).append(item)
        return {"accepted": accepted, "review_queue": review}

    return _split_results_node


def make_report_node(report_fn: ReportFn = _default_report_fn) -> Callable:
    @traceable(name="report_node", tags=make_trace_tags("report"))
    async def _report_node(state: BatchState) -> dict:
        report = await _call_maybe_async(report_fn, state.get("accepted", []), state.get("review_queue", []))
        return {"report": report}

    return _report_node


# ---------------------------------------------------------------------------
# Per-item nodes (Send-dispatched, async, semaphore-capped)
# ---------------------------------------------------------------------------


def make_extract_and_phase_a_node(
    extraction_fn: ExtractionFn = _default_extraction_fn,
    phase_a_fn: PhaseAFn = _default_phase_a_fn,
) -> Callable:
    """Send-fan-out #1 target. Extraction gets Class 2's inline schema-repair
    retry (cap 2). Phase A gets NO retry — a vector-store failure is Class
    3a, fail fast. Both run under the shared semaphore so 100 concurrent
    Sends don't all hit the API at once."""

    @traceable(name="extract_and_phase_a_node")
    async def _node(state: ItemState) -> dict:
        raw_row = state.get("raw_row", {})
        extraction_id = state.get("extraction_id", "unknown")

        normalized: dict | None = None
        last_error: Exception | None = None
        async with _SEMAPHORE:
            for attempt in range(1, 3):  # cap 2, §1.4 Class 2
                try:
                    prompt_input = raw_row if last_error is None else {**raw_row, "_repair_note": str(last_error)}
                    normalized = await _call_maybe_async(extraction_fn, prompt_input)
                    attach_trace_metadata(model_used="llama-3.1-8b-instant", extraction_id=extraction_id, attempt=str(attempt))
                    break
                except ValidationError as ve:
                    last_error = ve
                    logger.warning("extraction schema-repair retry %s/2 for %s: %s", attempt, extraction_id, ve)
                    continue
                except Exception as e:
                    raise classify_api_exception(e) from e

            if normalized is None:
                raise SchemaRepairExhaustedError(
                    f"extraction failed schema validation twice for {extraction_id}: {last_error}"
                )

            try:
                candidates = await _call_maybe_async(phase_a_fn, normalized)
            except Exception as e:
                classified = classify_api_exception(e)
                if not isinstance(classified, VectorStoreFatalError):
                    classified = VectorStoreFatalError(str(e))
                raise classified from e

        item = dict(state)
        item.update(normalized_product=normalized, candidates=candidates)
        return {"phase_a_out": [item]}

    return _node


def make_enrich_and_evaluate_node(
    enrichment_fn: EnrichmentFn = _default_enrichment_fn,
    judge_call_fn=None,  # if overridden: MUST STAY SYNC, same reason as _default_disambiguation_fn above — evaluate_item calls it directly, not awaited
    disambiguation_fn=None,
    low_cutoff: float = 0.35,
    high_cutoff: float = 0.65,
) -> Callable:
    """Send-fan-out #2 target for MATCHED_EXISTING ONLY (routing decision,
    locked in — NEW_PRODUCT and AMBIGUOUS go to evaluate_only_node instead).
    evaluate_item is synchronous (real Days 12-13 code) — bridged via
    _call_maybe_async so it runs in a thread, not blocking the loop."""

    @traceable(name="enrich_and_evaluate_node")
    async def _node(state: ItemState) -> dict:
        normalized = state.get("normalized_product") or {}
        candidates = state.get("candidates", [])
        catalog_id = (state.get("match_result") or {}).get("catalog_id", state.get("catalog_id", ""))
        extraction_id = state.get("extraction_id", "unknown")

        async with _SEMAPHORE:
            try:
                enrichment_raw = await _call_maybe_async(enrichment_fn, normalized, candidates)
            except Exception as e:
                raise classify_api_exception(e) from e

            enrichment_result = EnrichmentResult(
                catalog_id=enrichment_raw.get("catalog_id", catalog_id),
                field_results=[FieldEnrichment(**fr) for fr in enrichment_raw.get("field_results", [])],
            )
            match_result = MatchResult(**state["match_result"])

            kwargs = {}
            if judge_call_fn is not None:
                kwargs["judge_call_fn"] = judge_call_fn
            if disambiguation_fn is not None:
                kwargs["disambiguation_fn"] = disambiguation_fn

            record = await _call_maybe_async(
                evaluate_item,
                catalog_id=catalog_id,
                match_result=match_result,
                enrichment_result=enrichment_result,
                low_cutoff=low_cutoff,
                high_cutoff=high_cutoff,
                **kwargs,
            )
            attach_trace_metadata(model_used="gemini-2.5-flash", extraction_id=extraction_id)

        item = dict(state)
        item.update(enrichment_result=enrichment_raw, evaluation_record=record.model_dump())
        return {"eval_out": [item]}

    return _node


def make_evaluate_only_node(
    disambiguation_fn=_default_disambiguation_fn,
    low_cutoff: float = 0.35,
    high_cutoff: float = 0.65,
) -> Callable:
    """Send-fan-out #2 target for NEW_PRODUCT and AMBIGUOUS (routing
    decision, locked in). Skips Enrichment entirely — evaluates extraction
    only. Same node handles both decisions: evaluate_match() already only
    fires LLM disambiguation when decision == AMBIGUOUS, so NEW_PRODUCT
    passing through here just takes the plain-ACCEPT branch, no special-
    casing needed."""

    @traceable(name="evaluate_only_node")
    async def _node(state: ItemState) -> dict:
        catalog_id = (state.get("match_result") or {}).get("catalog_id", state.get("catalog_id", ""))
        extraction_id = state.get("extraction_id", "unknown")
        match_result = MatchResult(**state["match_result"])

        kwargs = {}
        if disambiguation_fn is not None:
            kwargs["disambiguation_fn"] = disambiguation_fn

        async with _SEMAPHORE:
            record = await _call_maybe_async(
                evaluate_item,
                catalog_id=catalog_id,
                match_result=match_result,
                enrichment_result=None,
                low_cutoff=low_cutoff,
                high_cutoff=high_cutoff,
                **kwargs,
            )
            attach_trace_metadata(model_used="llama-3.3-70b-versatile", extraction_id=extraction_id)

        item = dict(state)
        item.update(evaluation_record=record.model_dump())
        return {"eval_out": [item]}

    return _node