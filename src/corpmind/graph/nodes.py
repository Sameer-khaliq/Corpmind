"""
graph/nodes.py

CorpMind — Graph nodes (Day 14), rebuilt against the REAL state schema
==========================================================================

REBUILD CONTEXT — read this before touching anything else, because it
explains why this looks structurally different from earlier versions:

Every earlier version of this file (and edges.py/build_graph.py) invented
BatchState/ItemState field names (`phase_a_out`, `matched_items`, `eval_out`,
`feed_descriptor`, `report`, ...) because I never had the real
schemas/state.py. That mismatch was the actual root cause of the
`ainvoke() -> None` bug that took several rounds to chase down — LangGraph
was silently dropping state updates that wrote to keys that don't exist on
the real BatchState/ItemState TypedDicts. Once the real state.py surfaced,
the fix wasn't a patch, it was a rebuild: the real schema only has ONE
Send-accumulator key (`items`, `Annotated[list[ItemState], operator.add]`),
not one per pipeline stage.

That single-accumulator constraint, combined with the confirmed fact (from
schemas/matching.py: NEW_PRODUCT's catalog_id is "shared across an
intra-batch duplicate cluster") that Phase B connected-components
clustering genuinely needs ALL items' candidates at once — not per-item —
forces a different shape than the earlier two-Send-stage design:

    START
      │
      ▼
  extract_and_match (ONE batch-level async node — see below)
      │   internally: extraction (concurrent per row) -> Phase A candidate
      │   retrieval (concurrent per item) -> Phase B clustering (sequential,
      │   needs everyone's candidates at once) -> match_result assigned per
      │   item. Writes `items` ONCE, fully populated — never touched again,
      │   so the operator.add reducer never gets a chance to duplicate it.
      │
      ▼  route_after_matching (conditional edge, Send fan-out — the ONLY
      │  Send stage now, since matching itself isn't per-item dispatched)
      ├── MATCHED_EXISTING ──────► enrich_and_evaluate  ─┐
      └── NEW_PRODUCT/AMBIGUOUS ─► evaluate_only         ├─► each writes
                                                           │  directly into
                                                           │  accepted_items
                                                           │  or flagged_items
                                                           ▼
                                                          END (implicit join)

No separate "phase_b_matching", "split_results", or "report" node — none of
those keys exist on the real BatchState. Report generation, if wanted, is a
plain function over the final accepted_items/flagged_items/audit_log after
ainvoke() returns, not a graph node — see generate_report() at the bottom.

REAL, UNRESOLVED GAP — flagged, not glossed over: `accepted_items` is typed
`Annotated[list[ConsistentProduct], operator.add]` — items that ACCEPT need
converting from ItemState into a ConsistentProduct, presumably by a
"consistency agent" that hasn't come up in this build (Days 1-16 covered
ingestion/extraction/matching/enrichment/evaluation/graph/retry/tracing/
rate-limiting — nothing produces ConsistentProduct yet). I do NOT have that
schema, so `_default_consistency_fn` below raises NotImplementedError with a
clear message rather than fabricating a fake shape. Until it's wired, ANY
item that reaches an ACCEPT verdict will raise there — the Day 14 smoke
test below deliberately uses stub data that resolves to REJECT_TO_REVIEW so
it can still prove the wiring end-to-end without needing that agent to
exist yet. Send me schemas/consistent.py (ConsistentProduct) when you have
a minute and I'll wire this for real instead of stubbing around it.

WIRING YOU MUST DO:
  1. `_default_extraction_fn`, `_default_phase_a_fn`, `_default_phase_b_fn`,
     `_default_enrichment_fn`, `_default_consistency_fn` are ALL stubs.
     Wire real agents.* calls via each `make_*_node()` factory's params.
  2. `_default_consistency_fn` raises NotImplementedError until you share
     ConsistentProduct's real shape (see gap above).
"""

from __future__ import annotations

import asyncio
import inspect
import random
import time
from typing import Callable

from pydantic import ValidationError

from corpmind.graph.tracing_config import (
    SchemaRepairExhaustedError,
    VectorStoreFatalError,
    attach_trace_metadata,
    classify_api_exception,
    make_trace_tags,
    traceable,
)

try:
    import logging

    logger = logging.getLogger(__name__)
except ModuleNotFoundError:
    import logging

    logger = logging.getLogger(__name__)

    class _StubSettings:
        pass

    settings = _StubSettings()

# Real schema — confirmed from your actual schemas/state.py and
# schemas/matching.py uploads. No fallback mirror this time; if this import
# fails, that's a real problem to surface, not paper over with a guess.
try:
    from corpmind.schemas.state import BatchState, ItemState  # type: ignore
    from corpmind.schemas.matching import MatchDecision, MatchResult  # type: ignore
except ModuleNotFoundError:
    # Sandbox-only fallback (this exact shape, copied verbatim from your
    # uploads) so this file is still standalone-testable here.
    import operator
    from enum import Enum
    from typing import Annotated, TypedDict

    from pydantic import BaseModel, Field, model_validator

    class MatchDecision(str, Enum):
        NEW_PRODUCT = "NEW_PRODUCT"
        MATCHED_EXISTING = "MATCHED_EXISTING"
        AMBIGUOUS = "AMBIGUOUS"

    class MatchResult(BaseModel):
        catalog_id: str | None = Field(default=None)
        rrf_score: float
        decision: MatchDecision

        @model_validator(mode="after")
        def catalog_id_consistency(self) -> "MatchResult":
            needs_id = self.decision in (MatchDecision.MATCHED_EXISTING, MatchDecision.NEW_PRODUCT)
            if needs_id and self.catalog_id is None:
                raise ValueError(f"decision is {self.decision} but catalog_id is not set")
            if not needs_id and self.catalog_id is not None:
                raise ValueError(f"decision is {self.decision} but catalog_id is set")
            return self

    class ItemState(TypedDict, total=False):
        raw_row: dict
        normalized_product: dict | None
        match_result: dict | None
        enrichment_result: dict | None
        evaluation_record: dict | None
        consistent_output: dict | None
        audit_entries: list
        error: str | None

    class BatchState(TypedDict, total=False):
        batch_id: str
        supplier_feeds: list
        items: Annotated[list, operator.add]
        accepted_items: Annotated[list, operator.add]
        flagged_items: Annotated[list, operator.add]
        audit_log: Annotated[list, operator.add]

try:
    from corpmind.agents.evaluation import (  # type: ignore
        EnrichmentResult,
        FieldEnrichment,
        evaluate_item,
    )
except ModuleNotFoundError:
    from corpmind.agents.evaluation import (  # type: ignore  (sandbox fallback filename)
        EnrichmentResult,
        FieldEnrichment,
        evaluate_item,
    )

async def _call_maybe_async(fn: Callable, *args, **kwargs):
    if inspect.iscoroutinefunction(fn):
        return await fn(*args, **kwargs)
    return await asyncio.to_thread(fn, *args, **kwargs)


# ---------------------------------------------------------------------------
# Injectable hook stubs — every one of these needs real wiring
# ---------------------------------------------------------------------------

_STUB_LATENCY_SECONDS = 0.05


async def _default_extraction_fn(raw_row: dict) -> dict:
    # WIRING: replace with your real (ideally async) Groq extraction call.
    await asyncio.sleep(_STUB_LATENCY_SECONDS + random.uniform(0, 0.02))
    return {**raw_row, "field_provenance": {}, "extraction_warnings": []}


async def _default_phase_a_fn(normalized: dict) -> list[dict]:
    # WIRING: replace with your real (ideally async) vector-store candidate
    # lookup for ONE item. Called concurrently across items by
    # extract_and_match_node, capped by batch_runner.py's max_concurrency
    # (NOT by an internal semaphore here — see Day 16's design).
    await asyncio.sleep(_STUB_LATENCY_SECONDS)
    return []


async def _default_phase_b_fn(items_with_candidates: list[dict]) -> list[dict]:
    """
    WIRING: replace with your real connected-components clustering.
    Must run ONCE over the WHOLE batch (this is the cross-item join
    schemas/matching.py's "shared across an intra-batch duplicate cluster"
    comment confirms is real) — sequential, single-writer, not a
    concurrency target.

    Stub behavior (for smoke-testing the wiring, not real matching):
    assigns MATCHED_EXISTING to every 3rd item, AMBIGUOUS to every 5th,
    NEW_PRODUCT otherwise — NEW_PRODUCT items get a per-item catalog_id
    here (the stub doesn't simulate intra-batch clustering; your real
    implementation is where duplicate NEW_PRODUCT items must share one).
    """
    out = []
    for i, item in enumerate(items_with_candidates):
        item = dict(item)
        if i % 5 == 4:
            match_result = MatchResult(catalog_id=None, rrf_score=0.5, decision=MatchDecision.AMBIGUOUS)
        elif i % 3 == 2:
            match_result = MatchResult(catalog_id=f"cat-{i:04d}", rrf_score=0.9, decision=MatchDecision.MATCHED_EXISTING)
        else:
            match_result = MatchResult(catalog_id=f"cat-{i:04d}", rrf_score=0.9, decision=MatchDecision.NEW_PRODUCT)
        item["match_result"] = match_result
        out.append(item)
    return out


async def _default_enrichment_fn(normalized: dict, candidates: list[dict]) -> dict:
    # WIRING: replace with your real (ideally async) Tavily+LLM enrichment call.
    await asyncio.sleep(_STUB_LATENCY_SECONDS * 2)
    return {"catalog_id": normalized.get("catalog_id", ""), "field_results": []}


def _default_disambiguation_fn(match_result) -> dict:
    # MUST STAY SYNC — evaluate_item calls this directly, not awaited.
    time.sleep(_STUB_LATENCY_SECONDS)
    return {"resolved": True, "confidence": 0.9, "reasoning": "stub disambiguation for smoke testing"}


def _default_consistency_fn(item: ItemState) -> dict:
    """
    GAP, flagged plainly (see module docstring): converts an ACCEPT-verdict
    ItemState into a ConsistentProduct for accepted_items. I don't have
    ConsistentProduct's real schema, so this raises rather than fabricate
    one. Share schemas/consistent.py and I'll wire this for real.
    """
    raise NotImplementedError(
        "No consistency agent wired yet — ConsistentProduct's schema hasn't been shared. "
        "This item reached ACCEPT and needs converting from ItemState to ConsistentProduct "
        "before it can go into accepted_items. Wire a real consistency_fn, or share "
        "schemas/consistent.py so I can build the default."
    )


# ---------------------------------------------------------------------------
# Batch-level node: extraction + Phase A + Phase B, combined
# ---------------------------------------------------------------------------


def make_extract_and_match_node(
    extraction_fn=_default_extraction_fn,
    phase_a_fn=_default_phase_a_fn,
    phase_b_fn=_default_phase_b_fn,
    extraction_concurrency: int = 10,
) -> Callable:
    """
    ONE node covering extraction -> Phase A -> Phase B for the WHOLE batch.
    Writes `items` exactly once (see module docstring for why this can't be
    split across Send-dispatched branches the way it was in earlier,
    wrong-schema versions of this file).

    Extraction and Phase A run CONCURRENTLY across items (asyncio.gather,
    capped by extraction_concurrency here — this is a local cap distinct
    from batch_runner.py's graph-level max_concurrency, since this all
    happens inside a single node invocation, not via Send). Phase B runs
    sequentially after, since it needs everyone's candidates at once.
    """

    @traceable(name="extract_and_match_node", tags=make_trace_tags("extract_and_match"))
    async def _node(state: BatchState) -> dict:
        
        raw_rows = [item["raw_row"] for item in state.get("items", [])]  # WIRING: confirm real key — supplier_feeds is list[str] (file paths?) per your schema; adjust once ingestion.py's real contract is confirmed
        semaphore = asyncio.Semaphore(extraction_concurrency)

        async def extract_one(raw_row: dict) -> dict:
            async with semaphore:
                last_error: Exception | None = None
                for attempt in range(1, 3): 
                    try:
                        prompt_input = raw_row if last_error is None else {**raw_row, "_repair_note": str(last_error)}
                        normalized = await _call_maybe_async(extraction_fn, prompt_input)
                        return {"raw_row": raw_row, "normalized_product": normalized}
                    except ValidationError as ve:
                        last_error = ve
                        logger.warning("extraction schema-repair retry %s/2: %s", attempt, ve)
                        continue
                    except Exception as e:
                        raise classify_api_exception(e) from e
                raise SchemaRepairExhaustedError(f"extraction failed schema validation twice: {last_error}")

        extracted = await asyncio.gather(*[extract_one(row) for row in raw_rows])

        async def phase_a_one(item: dict) -> dict:
            async with semaphore:
                try:
                    candidates = await _call_maybe_async(phase_a_fn, item["normalized_product"])
                except Exception as e:
                    classified = classify_api_exception(e)
                    if not isinstance(classified, VectorStoreFatalError):
                        classified = VectorStoreFatalError(str(e))
                    raise classified from e
                return {**item, "candidates": candidates}

        with_candidates = await asyncio.gather(*[phase_a_one(item) for item in extracted])

        # Phase B: sequential, cross-item, needs everyone's candidates at once
        matched = await _call_maybe_async(phase_b_fn, with_candidates)

        items: list[ItemState] = [
            {"raw_row": m["raw_row"], "normalized_product": m["normalized_product"], "match_result": m["match_result"]}
            for m in matched
        ]
        attach_trace_metadata(model_used="llama-3.1-8b-instant", extraction_id=state.get("batch_id", "unknown"))
        return {"items": items}

    return _node


# ---------------------------------------------------------------------------
# Per-item nodes (Send-dispatched — the only Send stage now)
# ---------------------------------------------------------------------------


def make_enrich_and_evaluate_node(
    enrichment_fn=_default_enrichment_fn,
    judge_call_fn=None,
    disambiguation_fn=None,
    consistency_fn=_default_consistency_fn,
    low_cutoff: float = 0.35,
    high_cutoff: float = 0.65,
) -> Callable:
    """Send target for MATCHED_EXISTING only. Writes directly into
    accepted_items or flagged_items based on evaluation_record's verdict —
    both are operator.add accumulators, so appending once per item here is
    correct (not the same conflict `items` would have)."""

    @traceable(name="enrich_and_evaluate_node")
    async def _node(state: ItemState) -> dict:
        normalized = state.get("normalized_product") or {}
        match_result: MatchResult = state["match_result"]
        catalog_id = match_result.catalog_id

        try:
            enrichment_raw = await _call_maybe_async(enrichment_fn, normalized, [])
        except Exception as e:
            raise classify_api_exception(e) from e

        enrichment_result = EnrichmentResult(
            catalog_id=enrichment_raw.get("catalog_id", catalog_id),
            field_results=[FieldEnrichment(**fr) for fr in enrichment_raw.get("field_results", [])],
        )

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
        attach_trace_metadata(model_used="gemini-2.5-flash", extraction_id=catalog_id or "unknown")

        item: ItemState = {**state, "enrichment_result": enrichment_raw, "evaluation_record": record}

        if record.overall_verdict == "ACCEPT":
            consistent = await _call_maybe_async(consistency_fn, item)
            return {"accepted_items": [consistent]}
        return {"flagged_items": [item]}

    return _node


def make_evaluate_only_node(
    disambiguation_fn=_default_disambiguation_fn,
    consistency_fn=_default_consistency_fn,
    low_cutoff: float = 0.35,
    high_cutoff: float = 0.65,
) -> Callable:
    """Send target for NEW_PRODUCT and AMBIGUOUS — skips Enrichment
    entirely. Same node for both decisions: evaluate_match() only fires
    disambiguation when decision == AMBIGUOUS, so NEW_PRODUCT just takes
    the plain-ACCEPT branch there."""

    @traceable(name="evaluate_only_node")
    async def _node(state: ItemState) -> dict:
        match_result: MatchResult = state["match_result"]

        if match_result is None:
            from corpmind.schemas.evaluation import EvaluationRecord
            from corpmind.schemas.audit import AuditLogEntry
            import time

            fail_record = EvaluationRecord(
                catalog_id="unknown_structural_bypass",
                match_evaluation=None,
                field_evaluations={},
                overall_verdict="review",
                overall_reason="Pipeline structural failure: item reached evaluate_only_node without a valid match_result state."
            )
            
            err_audit = AuditLogEntry(
                catalog_id="unknown_structural_bypass",
                agent="evaluate_only_node",
                action="flagged_for_review",
                reason="Pipeline structural failure: item reached evaluate_only_node without a valid match_result state.",
                audit_tag="structural_bug",
            )
            
            # Append safely to state metrics tracking
            current_audit_entries = list(state.get("audit_entries", [])) or []
            current_audit_entries.append(err_audit)

            item: ItemState = {
                **state, 
                "evaluation_record": fail_record,
                "audit_entries": current_audit_entries
            }
            return {"flagged_items": [item]}

        catalog_id = match_result.catalog_id

        kwargs = {}
        if disambiguation_fn is not None:
            kwargs["disambiguation_fn"] = disambiguation_fn

        record = await _call_maybe_async(
            evaluate_item,
            catalog_id=catalog_id,
            match_result=match_result,
            enrichment_result=None,
            low_cutoff=low_cutoff,
            high_cutoff=high_cutoff,
            **kwargs,
        )
        attach_trace_metadata(model_used="llama-3.3-70b-versatile", extraction_id=catalog_id or "unknown")

        item: ItemState = {**state, "evaluation_record": record}

        if record.overall_verdict == "ACCEPT":
            consistent = await _call_maybe_async(consistency_fn, item)
            return {"accepted_items": [consistent]}
        return {"flagged_items": [item]}

    return _node


def generate_report(final_state: BatchState) -> dict:
    return {
        "accepted_count": len(final_state.get("accepted_items", [])),
        "flagged_count": len(final_state.get("flagged_items", [])),
        "audit_entries": len(final_state.get("audit_log", [])),
    }