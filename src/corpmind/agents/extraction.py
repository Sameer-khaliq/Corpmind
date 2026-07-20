"""
CorpMind — Extraction & Normalization Agent
"""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import ValidationError

from corpmind.config import settings
from corpmind.schemas.raw import RawProduct
from corpmind.schemas.extraction import FieldExtraction, NormalizedProduct

logger = logging.getLogger(__name__)

BATCH_SIZE = 10
MAX_REPROMPTS = 2  

EXTRACTABLE_FIELDS = (
    "title", "brand", "category", "color",
    "material", "size", "price", "sku", "description",
)

UNRESOLVED_TITLE_MARKER = "[UNRESOLVED — see extraction_warnings]"
UNRESOLVED_CATEGORY_MARKER = "needs_review"  

SYSTEM_PROMPT = (
    "You are a catalog data extraction engine. You will receive a JSON "
    "object with a `rows` array. Each row has a `source_row_index` and "
    "`raw_fields` (arbitrary supplier-provided fields, names and structure "
    "vary; some values may be null).\n\n"
    "For EVERY row, extract these fields: "
    + ", ".join(EXTRACTABLE_FIELDS) + ".\n\n"
    "Return ONLY a JSON object of the form:\n"
    '{"items": [{"source_row_index": <int>, '
    '"title": {"value": <string or null>, "confidence": 0.0-1.0}, '
    '"brand": {"value": ..., "confidence": ...}, ...}, ...]}\n\n'
    "Rules:\n"
    "- Return exactly one item per input row, in the same order.\n"
    "- `value` is a plain string or null. NEVER invent a value that isn't "
    "grounded in raw_fields.\n"
    "- `title` and `category` should almost always be extractable — flag "
    "low confidence rather than guessing if genuinely unclear.\n"
    "- Output must be valid JSON. No prose, no markdown fences."
)


def _build_user_prompt(rows: list[RawProduct]) -> str:
    payload = [
        {"source_row_index": r.source_row_index, "raw_fields": r.raw_fields}
        for r in rows
    ]
    return json.dumps({"rows": payload}, ensure_ascii=False)


def _call_llm(client: Any, messages: list[dict]) -> str:
    response = client.chat.completions.create(
        model=settings.extraction_model,
        messages=messages,
        temperature=0,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content


def _parse_response(
    raw_text: str, rows: list[RawProduct]
) -> tuple[dict[int, NormalizedProduct], str | None]:
    try:
        parsed = json.loads(raw_text)
        items = parsed["items"]
        if not isinstance(items, list):
            raise TypeError("`items` is not a list")
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        return {}, f"response was not valid JSON with an `items` array: {e}"

    if len(items) != len(rows):
        return {}, (
            f"expected {len(rows)} items, got {len(items)}. Every input row "
            "must produce exactly one output item, in the same order."
        )

    validated: dict[int, NormalizedProduct] = {}
    row_errors: list[str] = []

    for row, item in zip(rows, items):
        try:
            confidences: dict[str, float] = {}
            product_fields: dict[str, Any] = {}
            for field in EXTRACTABLE_FIELDS:
                field_data = item.get(field, {"value": None, "confidence": 0.0})
                fe = FieldExtraction.model_validate(field_data)
                confidences[field] = fe.confidence
                product_fields[field] = fe.value

            validated[row.source_row_index] = NormalizedProduct(
                supplier_id=row.supplier_id,
                source_row_index=row.source_row_index,
                field_confidences=confidences,
                **product_fields,
            )
        except ValidationError as e:
            
            row_errors.append(f"source_row_index {row.source_row_index}: {e.errors()!r}")

    if row_errors:
        return validated, "; ".join(row_errors)
    return validated, None


def _flagged_product(row: RawProduct, error: str | None) -> NormalizedProduct:
    """
    Terminal outcome for a row that never validated within MAX_REPROMPTS.

    Uses model_construct() deliberately — a normal constructor call would
    itself raise (title/category are required, category is taxonomy-gated).
    This is the ONLY place in this module that bypasses validation, and
    only because the whole point of the object is "this failed validation,
    a human needs to look at it" — never used for accepted/published data.
    """
    return NormalizedProduct.model_construct(
        supplier_id=row.supplier_id,
        source_row_index=row.source_row_index,
        title=UNRESOLVED_TITLE_MARKER,
        brand=None,
        category=UNRESOLVED_CATEGORY_MARKER,
        color=None,
        material=None,
        size=None,
        price=None,
        sku=None,
        description=None,
        field_confidences={f: 0.0 for f in EXTRACTABLE_FIELDS},
        extraction_warnings=[
            f"extraction failed after {MAX_REPROMPTS} schema-repair "
            f"reprompt(s): {error}"
        ],
    )


def extract_batch(client: Any, rows: list[RawProduct]) -> list[NormalizedProduct]:

    if not rows:
        return []

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _build_user_prompt(rows)},
    ]

    remaining = list(rows)
    validated: dict[int, NormalizedProduct] = {}
    last_error: str | None = None

    for attempt in range(MAX_REPROMPTS + 1):
        raw_text = _call_llm(client, messages)
        batch_validated, error = _parse_response(raw_text, remaining)
        validated.update(batch_validated)
        remaining = [r for r in remaining if r.source_row_index not in validated]

        if not remaining:
            break

        last_error = error
        if attempt == MAX_REPROMPTS:
            break  
        logger.warning(
            "extraction schema-repair reprompt %d/%d for %d row(s): %s",
            attempt + 1, MAX_REPROMPTS, len(remaining), last_error,
        )
        messages.append({"role": "assistant", "content": raw_text})
        messages.append({
            "role": "user",
            "content": (
                f"Your previous response was invalid: {last_error}\n"
                f"Return corrected JSON for ONLY these {len(remaining)} "
                f"row(s), same format as before: "
                + _build_user_prompt(remaining)
            ),
        })

    for row in remaining:
        validated[row.source_row_index] = _flagged_product(row, last_error)

    return [validated[r.source_row_index] for r in rows]


def run_extraction(client: Any, rows: list[RawProduct]) -> list[NormalizedProduct]:
    """Chunk into BATCH_SIZE-sized groups and extract each independently."""
    results: list[NormalizedProduct] = []
    for i in range(0, len(rows), BATCH_SIZE):
        chunk = rows[i : i + BATCH_SIZE]
        results.extend(extract_batch(client, chunk))
    return results
