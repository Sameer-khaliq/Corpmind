"""LangGraph state shapes, from §1.2 of the implementation plan.

These are TypedDicts, not Pydantic models — LangGraph's StateGraph reads
Annotated[..., reducer] metadata directly off TypedDict fields to know how
to merge parallel branches, which is how BatchState's accumulator fields
work below. The graph itself isn't wired until Day 14, but every node
written from Day 4 onward passes data shaped like this, so the contract
belongs with the rest of Day 2's schemas, not bolted on later.
"""
import operator
from typing import Annotated, TypedDict

from corpmind.schemas.audit import AuditLogEntry
from corpmind.schemas.consistent import ConsistentProduct
from corpmind.schemas.enrichment import EnrichmentResult
from corpmind.schemas.evaluation import EvaluationRecord
from corpmind.schemas.extraction import NormalizedProduct
from corpmind.schemas.matching import MatchResult
from corpmind.schemas.raw import RawProduct


class ItemState(TypedDict, total=False):
    """Per-item state. One of these is dispatched per RawProduct via
    LangGraph's Send() for parallel fan-out, and flows through Extraction ->
    Matching Phase A -> Enrichment -> Evaluation before rejoining the batch."""

    raw_row: RawProduct
    normalized_product: NormalizedProduct | None
    match_result: MatchResult | None
    enrichment_result: EnrichmentResult | None
    evaluation_record: EvaluationRecord | None
    consistent_output: ConsistentProduct | None
    audit_entries: list[AuditLogEntry]
    error: str | None


class BatchState(TypedDict, total=False):
    """Parent/batch-level state. Annotated[..., operator.add] fields are
    reducers — LangGraph appends each Send-dispatched branch's contribution
    instead of the last branch overwriting the others at the join."""

    batch_id: str
    supplier_feeds: list[str]
    items: Annotated[list[ItemState], operator.add]
    accepted_items: Annotated[list[ConsistentProduct], operator.add]
    flagged_items: Annotated[list[ItemState], operator.add]
    audit_log: Annotated[list[AuditLogEntry], operator.add]
