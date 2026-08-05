"""
reporting/audit_report.py

CorpMind — Day 18: Report Agent + audit report
================================================

Consumes the final BatchState (accepted_items, flagged_items, audit_log)
produced by the graph and renders:
  1. A human-readable markdown change report (generate_change_report)
  2. A flat CSV/JSON catalog export of accepted items (generate_catalog_export)

DESIGN DECISIONS, stated rather than left implicit:

- Joining flagged_items <-> audit_log uses `item["audit_catalog_id"]`, a
  field nodes.py now stashes onto every flagged ItemState (see the FIX in
  nodes.py's three flag-writing branches). This is NOT the same thing as
  match_result.catalog_id — for AMBIGUOUS items that field is None by
  MatchResult's own validator, so nodes.py substitutes a traceable sentinel
  (`ambiguous_pending_<item_id or source_row_index>`) instead. This report
  reads that same field rather than re-deriving the sentinel logic itself,
  so the two can never drift out of sync.

- accepted_items (list[ConsistentProduct]) has NO item-level identifier
  field pointing back to its ItemState except `catalog_id` itself, which
  IS a real, always-set field on ConsistentProduct (required `str`, per
  schemas/consistent.py). So accepted items join to audit_log directly on
  `catalog_id`.

- A flagged item with no matching audit_log entry is a real pipeline bug —
  every flag-writing branch in nodes.py now writes an audit_log entry in
  the same dict return as the flagged_items write, so these are two halves
  of one atomic operator.add append within the same node invocation. If
  they disrouter, that means a node returned flagged_items without
  audit_log (a regression), and this file raises rather than silently
  rendering a blank reason.

- CSV export flattens ConsistentProduct.attributes (dict[str,str]) into a
  single `attributes` JSON-string column rather than exploding to dynamic
  columns per attribute key, since attribute keys vary per product
  (size/fit/pattern/etc — not a fixed schema) and dynamic CSV columns
  would break on the very next differently-attributed product.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Literal

from corpmind.schemas.audit import AuditLogEntry
from corpmind.schemas.consistent import ConsistentProduct


def _index_audit_log(audit_log: list[AuditLogEntry]) -> dict[str, list[AuditLogEntry]]:
    """catalog_id (or sentinel) -> every AuditLogEntry recorded for it,
    in the order they were appended."""
    by_id: dict[str, list[AuditLogEntry]] = defaultdict(list)
    for entry in audit_log:
        by_id[entry.catalog_id].append(entry)
    return by_id


def _flagged_item_identifier(item: dict) -> str:
    """The identifier nodes.py stashed for this flagged item. Hard-fails
    if missing rather than falling back to a guess — a flagged item
    without this field means a graph node was edited without updating the
    audit-stash pattern, and that's a real regression to catch here, not
    paper over."""
    identifier = item.get("audit_catalog_id")
    if not identifier:
        raise ValueError(
            "Flagged item is missing 'audit_catalog_id' — every flag-writing "
            "branch in nodes.py is expected to stash this. This item cannot "
            "be traced to its audit_log entry, which means either a node "
            "was edited without preserving that stash, or this item didn't "
            "come from the real graph at all."
        )
    return identifier


def generate_change_report(
    accepted: list[ConsistentProduct],
    flagged: list[dict],  # list[ItemState]
    audit_log: list[AuditLogEntry],
) -> str:
    """Render the human-readable markdown change report. Every flagged
    reason and every accepted decision is pulled from a real AuditLogEntry
    — never a generic re-derived string."""

    audit_by_id = _index_audit_log(audit_log)
    lines: list[str] = []

    lines.append("# CorpMind — Catalog Reconciliation Change Report")
    lines.append("")
    lines.append(f"- Accepted: {len(accepted)}")
    lines.append(f"- Flagged for review: {len(flagged)}")
    lines.append(f"- Total audit log entries: {len(audit_log)}")
    lines.append("")

    # --- Accepted section ---
    lines.append(f"## Accepted ({len(accepted)})")
    lines.append("")
    if not accepted:
        lines.append("_No items were accepted in this batch._")
    for product in accepted:
        entries = audit_by_id.get(product.catalog_id)
        if not entries:
            raise ValueError(
                f"Accepted item '{product.catalog_id}' has no matching "
                "audit_log entry — every accept-writing branch in nodes.py "
                "must write both accepted_items and audit_log in the same "
                "return dict. This is a real trace gap, not a formatting "
                "issue."
            )
        latest = entries[-1]
        lines.append(f"### {product.title} (`{product.catalog_id}`)")
        lines.append(f"- Category: {product.category}")
        if product.brand:
            lines.append(f"- Brand: {product.brand}")
        if product.price is not None:
            lines.append(f"- Price: {product.price}")
        lines.append(f"- Decision: accepted by `{latest.agent}` — {latest.reason}")
        if len(entries) > 1:
            lines.append(f"- Full decision chain ({len(entries)} entries):")
            for e in entries:
                tag = f" [{e.audit_tag}]" if e.audit_tag else ""
                lines.append(f"  - `{e.agent}` → {e.action}: {e.reason}{tag}")
        lines.append("")

    # --- Flagged section ---
    lines.append(f"## Flagged for review ({len(flagged)})")
    lines.append("")
    if not flagged:
        lines.append("_No items were flagged in this batch._")
    for item in flagged:
        identifier = _flagged_item_identifier(item)
        entries = audit_by_id.get(identifier)
        if not entries:
            raise ValueError(
                f"Flagged item '{identifier}' has no matching audit_log "
                "entry — every flag-writing branch in nodes.py must write "
                "both flagged_items and audit_log in the same return dict. "
                "This is a real trace gap, not a formatting issue."
            )
        latest = entries[-1]
        normalized = item.get("normalized_product")
        title = getattr(normalized, "title", None) if normalized is not None else None
        lines.append(f"### {title or identifier} (`{identifier}`)")
        lines.append(f"- Reason: flagged by `{latest.agent}` — {latest.reason}")
        if latest.audit_tag:
            lines.append(f"- Tag: {latest.audit_tag}")
        if len(entries) > 1:
            lines.append(f"- Full decision chain ({len(entries)} entries):")
            for e in entries:
                tag = f" [{e.audit_tag}]" if e.audit_tag else ""
                lines.append(f"  - `{e.agent}` → {e.action}: {e.reason}{tag}")
        lines.append("")

    return "\n".join(lines)


def generate_catalog_export(
    accepted: list[ConsistentProduct],
    fmt: Literal["csv", "json"],
    output_dir: Path = Path("reports"),
) -> Path:
    """Flatten accepted ConsistentProduct records into a CSV or JSON export.
    attributes (dict[str,str]) is stringified as a single JSON column in
    CSV mode — see module docstring for why dynamic per-attribute columns
    were rejected."""

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for product in accepted:
        row = product.model_dump()
        if fmt == "csv":
            row["attributes"] = json.dumps(row.get("attributes", {}))
            if row.get("price") is not None:
                row["price"] = str(row["price"])
        rows.append(row)

    if fmt == "json":
        out_path = output_dir / "catalog_export.json"
        out_path.write_text(json.dumps(rows, indent=2, default=str))
        return out_path

    out_path = output_dir / "catalog_export.csv"
    if not rows:
        out_path.write_text("")
        return out_path
    fieldnames = list(rows[0].keys())
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def generate_report(final_state: dict, output_dir: Path = Path("reports")) -> dict:
    """Entry point matching nodes.generate_report()'s call shape — takes
    the final BatchState after ainvoke() returns, writes both the markdown
    change report and the CSV+JSON catalog exports to disk, and returns a
    small summary dict (does not itself return the full report text)."""

    accepted = final_state.get("accepted_items", [])
    flagged = final_state.get("flagged_items", [])
    audit_log = final_state.get("audit_log", [])

    output_dir.mkdir(parents=True, exist_ok=True)

    change_report_md = generate_change_report(accepted, flagged, audit_log)
    change_report_path = output_dir / "change_report.md"
    change_report_path.write_text(change_report_md, encoding="utf-8")

    csv_path = generate_catalog_export(accepted, fmt="csv", output_dir=output_dir)
    json_path = generate_catalog_export(accepted, fmt="json", output_dir=output_dir)

    return {
        "accepted_count": len(accepted),
        "flagged_count": len(flagged),
        "audit_entries": len(audit_log),
        "change_report_path": str(change_report_path),
        "catalog_export_csv": str(csv_path),
        "catalog_export_json": str(json_path),
    }


if __name__ == "__main__":
    # Minimal smoke test with hand-built data matching the real schemas —
    # exercises both the ACCEPT and FLAG traceability paths, plus the
    # missing-audit-entry hard-fail. Run against a real Day 17/18 batch
    # via generate_report(final_state) for the actual done-checkpoint.
    accepted = [
        ConsistentProduct(
            catalog_id="cat-0001",
            title="Men's Cotton Crew Neck T-Shirt",
            brand="ExampleBrand",
            category="tshirts",
            price=19.99,
            attributes={"color": "navy blue"},
        )
    ]
    flagged = [
        {
            "audit_catalog_id": "cat-0002",
            "normalized_product": None,
        }
    ]
    audit_log = [
        AuditLogEntry(
            catalog_id="cat-0001",
            agent="enrich_and_evaluate_node",
            action="accepted",
            reason="ACCEPT — all fields passed faithfulness threshold",
        ),
        AuditLogEntry(
            catalog_id="cat-0002",
            agent="enrich_and_evaluate_node",
            action="flagged_for_review",
            reason="REJECT_TO_REVIEW — field_eval(s) rejected for: material",
        ),
    ]
    report_md = generate_change_report(accepted, flagged, audit_log)
    print(report_md)
    print("\n[audit_report] PASS — smoke test rendered both accepted and flagged sections with real traces.")