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

FIXED (this revision) — confirmed 2x accepted+flagged bug: `extract_and_match_node`
was reading raw_rows FROM `state["items"]` and then writing its processed
output back into that SAME accumulator key. `items` is
`Annotated[list[ItemState], operator.add]`, so that write concatenated
instead of replacing — every batch silently doubled before routing even
ran (load-test logs confirmed exact 2x: 6 in -> 12 accepted+flagged, 8 in
-> 16). Fix: this node now reads from a dedicated `raw_rows` field (add it
to schemas/state.py as a plain, non-accumulator field if not already
there — ingestion must seed THAT, not `items`) and asserts `items` is
empty on entry, raising loudly instead of silently doubling if that
assumption is ever violated again.

FIXED (this revision) — `_default_consistency_fn` no longer hard-raises
NotImplementedError on every ACCEPT verdict. It now attempts a real
best-effort merge (NormalizedProduct fields overlaid with FILLED_GROUNDED
enrichment values) and constructs ConsistentProduct from that. Still
flagged: the exact ConsistentProduct field names haven't been confirmed
against this file, so a real schema mismatch will surface as a
ValidationError naming the bad field — that's expected until
schemas/consistent.py is shared and reconciled directly.

WIRING YOU MUST DO:
  1. `_default_extraction_fn`, `_default_phase_a_fn`, `_default_phase_b_fn`,
     `_default_enrichment_fn` are still stubs. Wire real agents.* calls via
     each `make_*_node()` factory's params.
  2. Add `raw_rows: list[dict]` to schemas/state.py's BatchState (plain
     field, not operator.add) and point ingestion's seeding at it instead
     of `items`, if that isn't already the case.
  3. Confirm `_default_consistency_fn`'s merge payload actually matches
     ConsistentProduct's real field names — run it once and read whatever
     ValidationError comes back.
"""

from __future__ import annotations

import asyncio
import inspect
import random
import time
import uuid
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


async def _default_enrichment_fn(normalized: dict) -> dict:
    # WIRING: replace with your real (ideally async) Tavily+LLM enrichment call
    # — corpmind.agents.enrichment.enrich_product(product), which takes a
    # NormalizedProduct model and returns an EnrichmentResult (see the
    # isinstance() normalization at the call site below for that boundary).
    await asyncio.sleep(_STUB_LATENCY_SECONDS * 2)
    return {"catalog_id": normalized.get("catalog_id", ""), "field_results": []}


def _default_disambiguation_fn(match_result) -> dict:
    # MUST STAY SYNC — evaluate_item calls this directly, not awaited.
    time.sleep(_STUB_LATENCY_SECONDS)
    return {"resolved": True, "confidence": 0.9, "reasoning": "stub disambiguation for smoke testing"}


def _default_consistency_fn(item: ItemState):
    """
    BEST-EFFORT WIRING, still flagged: schemas/consistent.py's real
    ConsistentProduct model has not been confirmed against this file, so
    this builds the merge payload from what IS confirmed — NormalizedProduct's
    real fields (extraction.py) overlaid with FieldEnrichment's
    FILLED_GROUNDED values (enrichment.py) — and lets ConsistentProduct
    itself validate the result. If the real model's field names differ,
    this raises a real, traceable pydantic ValidationError naming exactly
    which field is wrong, instead of either fabricating a fake shape or
    hard-blocking every ACCEPT verdict with NotImplementedError like the
    previous version did. Replace with a real
    corpmind.agents.consistency.build_consistent_product() once that agent
    exists — this is a stopgap so accepted_items can actually populate.
    """
    normalized = dict(item.get("normalized_product") or {})
    match_result = item.get("match_result")
    enrichment_result = item.get("enrichment_result") or {}

    payload = dict(normalized)

    # CRASH FIX: an AMBIGUOUS item that disambiguation resolves to ACCEPT
    # still has match_result.catalog_id == None -- MatchResult's own
    # validator forbids setting it while decision == AMBIGUOUS, so there is
    # no real catalog_id anywhere on match_result for this case, only the
    # traceable "ambiguous_pending_*" sentinel stashed elsewhere. Without
    # this override, payload never gets a catalog_id at all and
    # ConsistentProduct rejects it as a required field -- confirmed real
    # risk once real disambiguation was wired in (previously unreachable
    # with the stub). The caller (evaluate_only_node) mints a real
    # catalog_id and stashes it here before calling this function; normal
    # (non-ambiguous) items simply won't have this key set, falling back
    # to match_result.catalog_id as before.
    catalog_id = item.get("resolved_catalog_id")
    if catalog_id is None:
        catalog_id = getattr(match_result, "catalog_id", None)
        if catalog_id is None and isinstance(match_result, dict):
            catalog_id = match_result.get("catalog_id")
    if catalog_id:
        payload["catalog_id"] = catalog_id

    for fr in enrichment_result.get("field_results", []):
        if fr.get("resolution") == "FILLED_GROUNDED":
            payload[fr["field_name"]] = fr.get("enriched_value")

    try:
        from corpmind.schemas.consistent import ConsistentProduct  # type: ignore

        return ConsistentProduct(**payload)
    except ModuleNotFoundError:
        # Sandbox-only: real schema module isn't importable here. Return
        # the merged dict so the graph doesn't crash in this environment;
        # on the real machine the import above will succeed and this
        # branch never runs.
        return payload
    except Exception as e:
        raise ValueError(
            "consistency_fn built a merge payload but ConsistentProduct "
            f"rejected it — this is a real schema mismatch, not a wiring "
            f"bug, and needs the field names reconciled: {e}"
        ) from e


# ---------------------------------------------------------------------------
# Batch-level node: extraction + Phase A + Phase B, combined
# ---------------------------------------------------------------------------


def make_extract_and_match_node(
    extraction_fn=_default_extraction_fn,
    phase_a_fn=_default_phase_a_fn,
    phase_b_fn=_default_phase_b_fn,
    prepare_batch_index_fn=None,
    write_new_products_fn=None,
    extraction_concurrency: int = 10,
) -> Callable:
    """
    ONE node covering extraction -> Phase A -> Phase B for the WHOLE batch.
    Writes `items` exactly once (see module docstring for why this can't be
    split across Send-dispatched branches the way it was in earlier,
    wrong-schema versions of this file).

    Extraction runs CONCURRENTLY across items (asyncio.gather, capped by
    extraction_concurrency here — this is a local cap distinct from
    batch_runner.py's graph-level max_concurrency, since this all happens
    inside a single node invocation, not via Send).

    DUAL MODE, flagged rather than silently picked for you:
    - Default (prepare_batch_index_fn=None): stub-compatible mode. phase_a_fn
      takes ONE arg (normalized_product), phase_b_fn takes the whole
      with_candidates list and returns a LIST of per-item dicts. This is what
      the Day 14 smoke test and _default_phase_a_fn/_default_phase_b_fn
      still exercise — unchanged.
    - Real mode (prepare_batch_index_fn given — e.g.
      corpmind.agents.matching.prepare_batch_index): the REAL matching.py
      functions have a different contract than the stub: batch_index needs
      every item's embedding, so it can only be built ONCE, AFTER all
      extraction finishes — not per item, and not concurrently with
      extraction the way the stub's phase_a_fn was. phase_a_fn here is
      called as phase_a_fn(normalized_product, batch_index) (matching
      find_candidates_for_item's real signature), and phase_b_fn is called
      as phase_b_fn(all_candidate_pairs, normalized_products) and is
      expected to return dict[item_id, MatchResult] (matching
      resolve_batch's real return type — a dict, not a list, since a dict
      keyed by item_id cannot itself hold more than one MatchResult per
      item, structurally). Mapped back onto items via one dict.get() per
      item over a fixed-length zip — never list concatenation from two
      separate sources, which is the shape most likely to silently double
      an item if a future edit gets it wrong.
    """

    @traceable(name="extract_and_match_node", tags=make_trace_tags("extract_and_match"))
    async def _node(state: BatchState) -> dict:

        # BUG FIX (was the confirmed 2x accepted+flagged bug — 6 items in ->
        # 12 out, 8 items in -> 16 out, exact 2x every run): this node used
        # to build raw_rows by reading state["items"], then return a fully
        # processed `items` list back into that SAME key. `items` is an
        # operator.add accumulator, so that write does not replace — it
        # concatenates onto whatever state["items"] already held (the raw
        # rows this node just read), silently doubling every item before
        # routing even runs.
        #
        # Fix: this node is the ONLY writer of `items`, full stop, and it
        # must never read `items` as its own input. Raw rows now come from
        # `raw_rows` (a plain, non-accumulator BatchState field — add this
        # to schemas/state.py if not already there; ingestion must seed
        # THAT field, not `items`, at invoke time). If `items` is non-empty
        # when this node starts, that means something upstream is still
        # seeding the accumulator directly — fail loudly here instead of
        # silently doubling the batch again.
        existing_items = state.get("items", [])
        if existing_items:
            raise RuntimeError(
                f"extract_and_match_node started with {len(existing_items)} "
                "pre-existing item(s) already in state['items']. This node "
                "must be the sole writer of that operator.add accumulator — "
                "seeding raw rows into `items` upstream (ingestion or the "
                "invoke() call) is exactly what caused the 2x "
                "accepted+flagged bug. Seed raw rows into `raw_rows` "
                "instead and leave `items` empty until this node returns."
            )

        raw_rows = [row["raw_row"] if "raw_row" in row else row for row in state.get("raw_rows", [])]
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

        if prepare_batch_index_fn is None:
            # --- stub-compatible mode, unchanged from before ---
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
            matched = await _call_maybe_async(phase_b_fn, with_candidates)

            items: list[ItemState] = [
                {"raw_row": m["raw_row"], "normalized_product": m["normalized_product"], "match_result": m["match_result"]}
                for m in matched
            ]
        else:
            # --- real matching.py mode ---
            normalized_products = [e["normalized_product"] for e in extracted]
            batch_index = await _call_maybe_async(prepare_batch_index_fn, normalized_products)

            async def phase_a_one(e: dict) -> dict:
                async with semaphore:
                    try:
                        candidates = await _call_maybe_async(phase_a_fn, e["normalized_product"], batch_index)
                    except Exception as ex:
                        classified = classify_api_exception(ex)
                        if not isinstance(classified, VectorStoreFatalError):
                            classified = VectorStoreFatalError(str(ex))
                        raise classified from ex
                    return {**e, "candidate_pairs": candidates}

            with_candidates = await asyncio.gather(*[phase_a_one(e) for e in extracted])
            all_pairs = [p for e in with_candidates for p in e.get("candidate_pairs", [])]

            match_results: dict = await _call_maybe_async(phase_b_fn, all_pairs, normalized_products)

            if write_new_products_fn is not None:
                await _call_maybe_async(write_new_products_fn, normalized_products, match_results)

            # One dict lookup per item, over a fixed-length zip of the
            # original extracted list — the length of `items` is guaranteed
            # equal to len(extracted), so this cannot produce more entries
            # than items that went in.
            items = [
                {
                    "raw_row": e["raw_row"],
                    "normalized_product": e["normalized_product"],
                    "match_result": match_results.get(str(product.item_id)),
                }
                for e, product in zip(with_candidates, normalized_products)
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

        # WIRING NOTE, flagged rather than guessed silently: the stub
        # _default_enrichment_fn(normalized, candidates) takes two args, but
        # the real corpmind.agents.enrichment.enrich_product(product) takes
        # ONE — a NormalizedProduct model, not a dict, and it computes its
        # own missing-field list internally rather than taking `candidates`.
        # `candidates` was never consumed by the real enrichment agent per
        # the schemas/enrichment.py you shared, so it's dropped here rather
        # than passed to a function that can't use it. If real wiring needs
        # match-candidate context inside enrichment after all, that's a
        # signature change to enrich_product itself, not something to paper
        # over at the call site.
        try:
            enrichment_raw = await _call_maybe_async(enrichment_fn, normalized)
        except Exception as e:
            raise classify_api_exception(e) from e

        # enrichment_fn may return either a plain dict (stub contract) or a
        # real EnrichmentResult pydantic model (the real enrich_product) —
        # normalize to the dict shape this node already expects below.
        if isinstance(enrichment_raw, EnrichmentResult):
            enrichment_raw = enrichment_raw.model_dump()

        enrichment_result = EnrichmentResult(
            catalog_id=enrichment_raw.get("catalog_id", catalog_id),
            field_results=[FieldEnrichment(**fr) for fr in enrichment_raw.get("field_results", [])],
        )

        kwargs = {}
        if judge_call_fn is not None:
            kwargs["judge_call_fn"] = judge_call_fn
        if disambiguation_fn is not None:
            kwargs["disambiguation_fn"] = disambiguation_fn
        # FIX: extraction_warnings was never passed to evaluate_item, so a
        # structurally-failed extraction (schema-repair exhausted) had
        # nothing in the gate capable of catching it — see aggregate_verdict
        # in evaluation.py for the full fix. `normalized` may be a
        # NormalizedProduct object or (fallback) a dict — handle both.
        extraction_warnings = (
            getattr(normalized, "extraction_warnings", None)
            if not isinstance(normalized, dict)
            else normalized.get("extraction_warnings")
        )
        if extraction_warnings:
            kwargs["extraction_warnings"] = extraction_warnings

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

        item: ItemState = {
            **state,
            "enrichment_result": enrichment_raw,
            "evaluation_record": record,
            "audit_catalog_id": catalog_id,
        }

        # AUDIT FIX: previously this branch returned only accepted_items/
        # flagged_items — the batch-level `audit_log` accumulator was never
        # written here, so an accepted item's decision trail (why it was
        # accepted, by which agent) was lost the moment it left this node.
        # Day 18's report needs this to trace flagged reasons and accepted
        # audit trails back to a real AuditLogEntry, not a re-derived guess.
        from corpmind.schemas.audit import AuditLogEntry

        audit_entry = AuditLogEntry(
            catalog_id=catalog_id,
            agent="enrich_and_evaluate_node",
            action="accepted" if record.overall_verdict == "ACCEPT" else "flagged_for_review",
            reason=record.overall_reason,
            audit_tag="evaluation_verdict",
        )

        if record.overall_verdict == "ACCEPT":
            consistent = await _call_maybe_async(consistency_fn, item)
            return {"accepted_items": [consistent], "audit_log": [audit_entry]}
        return {"flagged_items": [item], "audit_log": [audit_entry]}

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
            from corpmind.agents.evaluation import EvaluationRecord, MatchEvalScore, aggregate_verdict
            from corpmind.schemas.audit import AuditLogEntry
            from corpmind.schemas.matching import MatchDecision

            # WIRING FIX: the previous version of this fallback used field
            # names (match_evaluation, field_evaluations) and a verdict
            # string ("review") that don't match the real EvaluationRecord /
            # MatchEvalScore schemas in evaluation.py — it would raise a
            # ValidationError the instant a structural failure actually hit
            # it, i.e. it crashed exactly when it was supposed to be the
            # safety net. Built via aggregate_verdict() so overall_verdict is
            # guaranteed consistent with match_eval, matching how
            # EvaluationRecord's own model_validator checks it everywhere else.
            structural_failure_match_eval = MatchEvalScore(
                catalog_id=None,
                rrf_score=0.0,
                decision=MatchDecision.AMBIGUOUS,  # sentinel — no real decision was made
                confidence=0.0,
                verdict="REJECT_TO_REVIEW",
                reason="Pipeline structural failure: item reached evaluate_only_node without a valid match_result state.",
                disambiguation_used=False,
            )
            overall_verdict, overall_reason = aggregate_verdict(structural_failure_match_eval, [])

            fail_record = EvaluationRecord(
                catalog_id="unknown_structural_bypass",
                match_eval=structural_failure_match_eval,
                field_evals=[],
                overall_verdict=overall_verdict,
                overall_reason=overall_reason,
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
                "audit_entries": current_audit_entries,
                "audit_catalog_id": "unknown_structural_bypass",
            }
            # AUDIT FIX: err_audit was previously stashed only into the
            # item's own `audit_entries` list, never surfaced to
            # BatchState's `audit_log` accumulator — so this structural
            # failure was invisible to the batch-level audit trail Day 18's
            # report reads from.
            return {"flagged_items": [item], "audit_log": [err_audit]}

        catalog_id = match_result.catalog_id
        normalized = state.get("normalized_product")

        # CRASH FIX: MatchResult's own validator requires catalog_id to be
        # None specifically when decision == AMBIGUOUS — this node handles
        # both NEW_PRODUCT (catalog_id always set) and AMBIGUOUS
        # (catalog_id always None). AuditLogEntry.catalog_id is a required
        # `str`, not Optional, so passing None straight through crashes with
        # a ValidationError on every AMBIGUOUS item. Fall back to a
        # traceable sentinel built from whatever item identifier is
        # available, so the audit entry still records something real
        # instead of either crashing or silently dropping the item.
        audit_catalog_id = catalog_id
        if audit_catalog_id is None:
            fallback_id = (
                getattr(normalized, "item_id", None) or getattr(normalized, "source_row_index", None)
                if normalized is not None
                else None
            )
            audit_catalog_id = f"ambiguous_pending_{fallback_id if fallback_id is not None else 'unknown'}"

        kwargs = {}
        if disambiguation_fn is not None:
            kwargs["disambiguation_fn"] = disambiguation_fn
        # FIX: this is the exact path the "[UNRESOLVED — see
        # extraction_warnings]" item went through and got silently
        # ACCEPTed — NEW_PRODUCT/AMBIGUOUS items skip Enrichment entirely,
        # so extraction_warnings was the ONLY signal available that
        # something was structurally wrong, and it was never read. See
        # aggregate_verdict in evaluation.py for the enforcement side.
        extraction_warnings = (
            getattr(normalized, "extraction_warnings", None)
            if not isinstance(normalized, dict)
            else (normalized or {}).get("extraction_warnings")
        )
        if extraction_warnings:
            kwargs["extraction_warnings"] = extraction_warnings

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

        item: ItemState = {**state, "evaluation_record": record, "audit_catalog_id": audit_catalog_id}

        # AUDIT FIX: same gap as enrich_and_evaluate_node — write a real
        # AuditLogEntry into the batch-level audit_log accumulator here too,
        # not just accepted_items/flagged_items.
        from corpmind.schemas.audit import AuditLogEntry

        resolved_catalog_id = audit_catalog_id
        if record.overall_verdict == "ACCEPT" and catalog_id is None:
            # CRASH FIX: this is the AMBIGUOUS-resolved-to-ACCEPT case (see
            # _default_consistency_fn's docstring above for the full why).
            # Mint a real catalog_id now and use THE SAME id for both the
            # accepted ConsistentProduct and this audit entry (not the
            # "ambiguous_pending_*" sentinel) — report.py's accepted-item
            # join is by product.catalog_id, so the audit entry must match
            # that exactly, not the pre-resolution sentinel.
            resolved_catalog_id = f"CM-{uuid.uuid4().hex[:12]}"
            item = {**item, "resolved_catalog_id": resolved_catalog_id}

        audit_entry = AuditLogEntry(
            catalog_id=resolved_catalog_id,
            agent="evaluate_only_node",
            action="accepted" if record.overall_verdict == "ACCEPT" else "flagged_for_review",
            reason=record.overall_reason,
            audit_tag="evaluation_verdict",
        )

        if record.overall_verdict == "ACCEPT":
            consistent = await _call_maybe_async(consistency_fn, item)
            return {"accepted_items": [consistent], "audit_log": [audit_entry]}
        return {"flagged_items": [item], "audit_log": [audit_entry]}

    return _node


def generate_report(final_state: BatchState) -> dict:
    return {
        "accepted_count": len(final_state.get("accepted_items", [])),
        "flagged_count": len(final_state.get("flagged_items", [])),
        "audit_entries": len(final_state.get("audit_log", [])),
    }