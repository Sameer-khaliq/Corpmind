"""
pipeline_adapter.py
Thin adapter between the Streamlit UI and CorpMind's real pipeline code.
app.py never imports corpmind internals directly except through this file --
if a real function signature differs from what's assumed here, fix it in
ONE place instead of hunting through app.py.

Every integration point below (ingestion, build_graph, report generation,
ConsistentProduct/AuditLogEntry field mapping) has been checked against
the real files, not guessed.
"""

import asyncio
import queue
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from corpmind.agents.ingestion import ingest_supplier_feed  # Day 3 -- confirmed real signature
from corpmind.graph.build_graph import build_graph        # Days 14-16
# Day 18 -- confirmed real shape: generate_change_report/generate_catalog_export
# take (accepted, flagged, audit_log) separately and generate_catalog_export
# WRITES A FILE and returns a Path, it doesn't return CSV/JSON text. Use the
# generate_report(final_state, output_dir) entry point instead -- it's the one
# report.py itself says matches "nodes.generate_report()'s call shape",
# i.e. takes the whole final BatchState and writes all three files at once.
from corpmind.agents.report import generate_report as generate_audit_report
from corpmind.schemas.audit import AuditLogEntry
from corpmind.schemas.raw import RawProduct
from corpmind.config import settings

# Real Day 4-13 agents -- build_graph()'s kwargs are ALL optional and fall
# back to nodes.py's smoke-test stubs if not passed. Without this wiring,
# the pipeline silently ran on stubs the whole time (confirmed: the
# "title Field required" crash was _default_extraction_fn's stub output,
# which only copies raw_fields through and never actually extracts
# title/category -- not a ConsistentProduct bug at all).
from groq import Groq
from corpmind.agents.extraction import extract_batch
from corpmind.agents.matching import (
    prepare_batch_index,
    find_candidates_for_item,
    resolve_batch,
    write_new_products,  
)
from corpmind.agents.enrichment import enrich_product
from corpmind.agents.evaluation import default_disambiguation_fn, default_judge_call_fn


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

def ingest_uploaded_files(uploaded_files, supplier_ids):
    """
    uploaded_files: list of Streamlit UploadedFile objects (in-memory,
                    file-like -- ingest_supplier_feed wants a real path
                    on disk, since it does path.suffix / path.exists() /
                    path.read_bytes() checks).
    supplier_ids:   list of str, same length, one per file.
    Returns: list[dict], flattened across all uploaded feeds.

    ingest_supplier_feed returns RawProduct pydantic objects, but
    schemas/state.py's own BatchState declares raw_rows: list[dict], and
    downstream code (extraction_fn, nodes.py's "raw_row" in row checks)
    treats each entry as a mapping -- `**row` / `row[...]` / `in row` all
    require a real dict, and a bare RawProduct object fails all three
    ("'RawProduct' object is not a mapping"). model_dump() each row here
    so raw_rows matches the type BatchState actually declares.

    Each uploaded file is spooled to a NamedTemporaryFile with its
    original suffix preserved (ingest_supplier_feed dispatches on
    path.suffix.lower() to decide CSV vs XLSX parsing) and cleaned up
    after ingestion, success or failure.
    """
    raw_rows = []
    for file, supplier_id in zip(uploaded_files, supplier_ids):
        suffix = Path(file.name).suffix.lower()
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(file.getvalue())
            tmp_path = Path(tmp.name)

        try:
            rows = ingest_supplier_feed(tmp_path, supplier_id=supplier_id)
            raw_rows.extend(row.model_dump() for row in rows)
        finally:
            tmp_path.unlink(missing_ok=True)

    return raw_rows


def _generate_reports(result_state, output_dir: Path | None = None):
    """
    Wraps the real generate_report(final_state, output_dir) -- it writes
    change_report.md + catalog_export.csv + catalog_export.json to disk
    and returns a dict of paths (accepted_count/flagged_count/etc, not the
    file contents). We read those files back so the UI can show/download
    them without the caller needing to know they're on disk at all.
    Returns (report_md: str, catalog_csv: bytes, catalog_json: bytes, report_dir: str).
    """
    out_dir = output_dir or Path(tempfile.mkdtemp(prefix="corpmind_report_"))
    summary = generate_audit_report(result_state, output_dir=out_dir)

    report_md = Path(summary["change_report_path"]).read_text(encoding="utf-8")
    catalog_csv = Path(summary["catalog_export_csv"]).read_bytes()
    catalog_json = Path(summary["catalog_export_json"]).read_bytes()
    return report_md, catalog_csv, catalog_json, str(out_dir)


def _build_consistent_product(item, catalog_id_override: str | None = None):
    """
    Builds a ConsistentProduct from a flagged ItemState -- needed for
    Day 20's approve action, mirroring but NOT literally reusing
    corpmind.graph.nodes._default_consistency_fn (private, and it has a
    real field-mapping bug now that we have the actual schemas -- see
    below).

    Why not just call _default_consistency_fn:
    1. MatchResult's own model_validator FORBIDS catalog_id being set at
       all when decision == AMBIGUOUS. An AMBIGUOUS item a human now
       approves has no real catalog_id anywhere on match_result --
       structurally absent, not "None we can patch" -- so we need to pass
       one in directly instead of reading it off match_result.
    2. nodes.py's version does `payload[fr["field_name"]] = enriched_value`
       for every FILLED_GROUNDED enrichment field, straight onto the
       top-level payload dict. That's fine when field_name is "title" or
       "description" (real ConsistentProduct columns), but color/material/
       size (both NormalizedProduct's own fields AND common enrichment
       targets) have NO matching top-level field on ConsistentProduct --
       they belong in `attributes: dict[str,str]`. Under pydantic's
       default extra="ignore", passing them at the top level doesn't
       error, it just silently drops the data. Fixed here by routing
       anything that isn't a real ConsistentProduct column into
       `attributes` instead.
    """
    from corpmind.schemas.consistent import ConsistentProduct

    normalized_obj = item.get("normalized_product")
    normalized = normalized_obj.model_dump() if normalized_obj is not None else {}
    match_result = item.get("match_result")
    enrichment_result = item.get("enrichment_result")

    payload = {
        "title": normalized.get("title"),
        "brand": normalized.get("brand"),
        "category": normalized.get("category"),
        "description": normalized.get("description"),
        "sku": normalized.get("sku"),
        "price": normalized.get("price"),
    }
    attributes: dict[str, str] = {}
    for extra_field in ("color", "material", "size"):
        val = normalized.get(extra_field)
        if val:
            attributes[extra_field] = val

    field_results = []
    if enrichment_result is not None:
        field_results = (
            enrichment_result.get("field_results", [])
            if isinstance(enrichment_result, dict)
            else getattr(enrichment_result, "field_results", [])
        )
    for fr in field_results:
        is_dict = isinstance(fr, dict)
        resolution = fr.get("resolution") if is_dict else fr.resolution
        if resolution != "FILLED_GROUNDED":
            continue
        field_name = fr.get("field_name") if is_dict else fr.field_name
        enriched_value = fr.get("enriched_value") if is_dict else fr.enriched_value
        if field_name in payload:
            payload[field_name] = enriched_value
        else:
            attributes[field_name] = enriched_value

    payload["attributes"] = attributes

    if catalog_id_override is not None:
        payload["catalog_id"] = catalog_id_override
    else:
        catalog_id = getattr(match_result, "catalog_id", None)
        if catalog_id is None and isinstance(match_result, dict):
            catalog_id = match_result.get("catalog_id")
        payload["catalog_id"] = catalog_id

    return ConsistentProduct(**payload)


def _make_extraction_fn():
    """
    nodes.py's extract_and_match_node calls extraction_fn(raw_row: dict)
    ONCE PER ITEM, concurrently (capped by extraction_concurrency). The
    real extract_batch(client, rows) takes a LIST of RawProduct and does
    its own internal batching (N=8-10/call, one shared system-prompt).

    Bridging these two contracts means calling extract_batch with a
    single-row list per item -- this pays full system-prompt overhead
    per item instead of amortizing it across a real batch, which is
    LESS efficient than Day 4's intended batching. Flagged, not silently
    "fixed" here: doing this properly means changing extract_and_match_node
    to call extraction over the whole raw_rows list at once instead of
    per-item, which is nodes.py's call shape to change, not this adapter's.
    Fine for a UI smoke-test batch; revisit before a real load test.
    """
    client = Groq(api_key=settings.GROQ_API_KEY)

    async def extraction_fn(raw_row: dict):
        # raw_row is RawProduct.model_dump() (see ingest_uploaded_files),
        # plus possibly a "_repair_note" key nodes.py adds on retry --
        # not a RawProduct field, strip it before reconstructing.
        clean = {k: v for k, v in raw_row.items() if k != "_repair_note"}
        raw_product = RawProduct(**clean)
        results = await asyncio.to_thread(extract_batch, client, [raw_product])
        return results[0]

    return extraction_fn


def _build_graph():
    """
    Wires the real Day 4-13 agents into build_graph(), instead of leaving
    every kwarg unset (which silently falls back to nodes.py's smoke-test
    stubs -- confirmed root cause of the earlier "title Field required"
    crash).

    phase_a_fn / prepare_batch_index_fn / phase_b_fn / write_new_products_fn
    are matching.py's real functions passed DIRECTLY -- nodes.py's own
    "real mode" docstring confirms their call shapes match exactly
    (phase_a_fn(normalized_product, batch_index), phase_b_fn(all_pairs,
    normalized_products) -> dict[item_id, MatchResult)), so no adapter
    needed for those.

    disambiguation_fn is explicitly wired to evaluation.py's REAL
    default_disambiguation_fn -- without this, nodes.py's own factory-level
    default (_default_disambiguation_fn, a stub) silently overrides
    evaluate_item's real default every time. judge_call_fn is passed
    explicitly too, even though evaluate_item's own default already points
    to the real default_judge_call_fn, just to remove any doubt about
    which default nodes.py's factory functions apply.
    """
    return build_graph(
        extraction_fn=_make_extraction_fn(),
        prepare_batch_index_fn=prepare_batch_index,
        phase_a_fn=find_candidates_for_item,
        phase_b_fn=resolve_batch,
        write_new_products_fn=write_new_products,
        enrichment_fn=enrich_product,
        judge_call_fn=default_judge_call_fn,
        disambiguation_fn=default_disambiguation_fn,
    ).compile()


# ---------------------------------------------------------------------------
# Pipeline execution with progress streaming -- ONE run, not two
# ---------------------------------------------------------------------------

def run_pipeline_in_background(raw_rows, progress_q: queue.Queue):
    """
    Runs the real LangGraph pipeline in a background thread, pushing a
    progress event onto progress_q as coarse stages complete. Final event
    is always either:
      {"type": "done", "result": <final BatchState dict>, "report_md": ...,
       "catalog_csv": ..., "catalog_json": ...}
    or
      {"type": "error", "error": str(exc)}
    """
    def _worker():
        try:
            asyncio.run(_run_async())
        except Exception as exc:  # noqa: BLE001
            progress_q.put({"type": "error", "error": str(exc)})

    async def _run_async():
        graph = _build_graph()

        initial_state = {
            "raw_rows": raw_rows,
            "items": [],
            "accepted_items": [],
            "flagged_items": [],
            "audit_log": [],
        }

        progress_q.put({
            "type": "stage_done",
            "stage": "ingestion",
            "label": "Ingestion",
            "detail": f"{len(raw_rows)} rows ingested",
            "ts": datetime.now(timezone.utc).isoformat(),
        })

        seen_extract_done = False
        prev_accepted = 0
        prev_flagged = 0
        final_state = initial_state

        # stream_mode="values" = fully merged BatchState snapshot after
        # every superstep (reducers already applied). Last snapshot IS the
        # final result -- no second run needed.
        async for state_snapshot in graph.astream(
            initial_state,
            config={"max_concurrency": settings.max_concurrent_llm_calls},  # confirmed = 10
            stream_mode="values",
        ):
            final_state = state_snapshot

            if not seen_extract_done and state_snapshot.get("items"):
                seen_extract_done = True
                progress_q.put({
                    "type": "stage_done",
                    "stage": "extract_and_match",
                    "label": "Extraction + Matching",
                    "detail": f"{len(state_snapshot['items'])} items processed",
                })

            n_accepted = len(state_snapshot.get("accepted_items", []))
            n_flagged = len(state_snapshot.get("flagged_items", []))
            if n_accepted != prev_accepted or n_flagged != prev_flagged:
                progress_q.put({
                    "type": "stage_done",
                    "stage": "review_processing",
                    "label": "Enrichment + Evaluation (per item)",
                    "detail": f"{n_accepted} accepted so far, {n_flagged} flagged so far",
                })
                prev_accepted, prev_flagged = n_accepted, n_flagged

        result_state = final_state
        report_md, catalog_csv, catalog_json, report_dir = _generate_reports(result_state)

        progress_q.put({
            "type": "done",
            "result": result_state,
            "report_md": report_md,
            "catalog_csv": catalog_csv,
            "catalog_json": catalog_json,
            "report_dir": report_dir,
        })

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    return thread


# ---------------------------------------------------------------------------
# Review queue actions (Day 20)
# ---------------------------------------------------------------------------

def approve_flagged_item(item, result_state, note: str = ""):
    """
    Moves a flagged ItemState into accepted_items and writes a
    REVIEWER_OVERRIDE audit entry. Mutates result_state (the session-state
    copy only -- does NOT re-run the graph).

    WATCH OUT: an AMBIGUOUS item that a human now approves has no real
    catalog_id anywhere -- MatchResult's own validator forbids setting one
    while decision == AMBIGUOUS. Detected via the "ambiguous_pending_*"
    sentinel nodes.py stashes as audit_catalog_id for exactly this case;
    we mint a fresh catalog_id here and pass it straight to
    _build_consistent_product, bypassing match_result entirely for that
    one field. Non-ambiguous flagged items (e.g. faithfulness-rejected)
    already have a real catalog_id on match_result, used as-is.
    """
    audit_catalog_id = item.get("audit_catalog_id", "")
    is_ambiguous_sentinel = audit_catalog_id.startswith("ambiguous_pending_")
    real_catalog_id = str(uuid.uuid4()) if is_ambiguous_sentinel else audit_catalog_id

    consistent_product = _build_consistent_product(item, catalog_id_override=real_catalog_id)

    result_state["accepted_items"].append(consistent_product)
    result_state["flagged_items"] = [
        f for f in result_state["flagged_items"]
        if f.get("audit_catalog_id") != audit_catalog_id
    ]

    audit_entry = AuditLogEntry(
        catalog_id=real_catalog_id,
        agent="human_reviewer",
        action="REVIEWER_OVERRIDE",
        reason="Approved by reviewer" + (f": {note}" if note else ""),
        audit_tag="human_override",
    )
    result_state["audit_log"].append(audit_entry)
    return result_state


def reject_flagged_item(item, result_state, note: str = ""):
    """
    Removes a flagged item from the active review list without adding it
    to the catalog, and logs a REVIEWER_OVERRIDE audit entry.
    """
    audit_catalog_id = item.get("audit_catalog_id", "")

    result_state["flagged_items"] = [
        f for f in result_state["flagged_items"]
        if f.get("audit_catalog_id") != audit_catalog_id
    ]

    audit_entry = AuditLogEntry(
        catalog_id=audit_catalog_id,
        agent="human_reviewer",
        action="REVIEWER_OVERRIDE",
        reason="Rejected by reviewer" + (f": {note}" if note else ""),
        audit_tag="human_override",
    )
    result_state["audit_log"].append(audit_entry)
    result_state.setdefault("rejected_items", []).append(item)
    return result_state


def regenerate_exports(result_state, report_dir: str | None = None):
    """
    Re-renders the change report + catalog export after a reviewer
    approve/reject action. Pass the report_dir from the original run (in
    session state as batch_result's sibling) to overwrite the same files
    in place rather than scattering a new temp dir per click.
    """
    out_dir = Path(report_dir) if report_dir else None
    return _generate_reports(result_state, output_dir=out_dir)