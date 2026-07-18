"""Extraction agent: turns a batch of RawProduct rows into NormalizedProduct
rows via a single LLM call per batch (happy path only — Day 5 owns retries
and validation-error recovery, this file assumes the call succeeds and the
model's JSON parses and validates cleanly).

Batching and prompt-caching design (§1.3):
  - One call handles N rows (recommended N=8-10 — see EXTRACTION_BATCH_SIZE),
    not one call per row. At ~450 requests/15min on llama-3.1-8b-instant's
    free-tier RPM alone, one-row-per-call can't hit the 500-SKU/15-min NFR
    before any other stage's calls stack on.
  - Groq's prompt caching is automatic and requires no API flag — it matches
    on a shared token PREFIX across consecutive requests. The only thing we
    control is prompt STRUCTURE: the system prompt below is byte-identical
    on every call (same instructions, same taxonomy list), and all per-batch
    variable content (the actual rows) lives in the user message, which
    comes after it. That ordering is what makes caching actually fire.
"""
import json
from decimal import Decimal, InvalidOperation

from groq import Groq
from pydantic import BaseModel, ValidationError

from corpmind.config import settings
from corpmind.schemas.extraction import FieldExtraction, FieldSource, NormalizedProduct
from corpmind.schemas.raw import RawProduct
from corpmind.taxonomy import load_taxonomy

EXTRACTION_BATCH_SIZE = 10  # recommended rows/call — callers should chunk to this

_FIELD_NAMES = (
    "title", "brand", "category", "color", "material", "size", "price", "sku", "description",
)


class _RawExtractionRow(BaseModel):
    """Shape we ask the LLM to emit for one row, before we translate it into
    a NormalizedProduct. Kept private to this module — it's a parsing
    contract for the LLM's JSON, not a cross-agent data contract."""

    row_index: int
    title: FieldExtraction
    brand: FieldExtraction
    category: FieldExtraction
    color: FieldExtraction
    material: FieldExtraction
    size: FieldExtraction
    price: FieldExtraction
    sku: FieldExtraction
    description: FieldExtraction


class _RawExtractionBatch(BaseModel):
    rows: list[_RawExtractionRow]


def build_system_prompt() -> str:
    """Static across every call — this is the cacheable prefix. Must not
    contain anything batch-specific (no row data, no batch size, no
    timestamps) or every call becomes a fresh cache miss."""
    valid_categories = sorted(load_taxonomy())
    field_list = ", ".join(_FIELD_NAMES)

    return f"""You are a product catalog extraction agent for an e-commerce
reconciliation pipeline. You will be given a batch of raw supplier product
rows. Each row has an arbitrary set of column names and raw string values —
different suppliers use completely different column names for the same
concept, and some rows are missing columns entirely.

For each row, extract these fields: {field_list}.

For EVERY field, return an object with exactly these three keys:
  "value": the extracted string value, or null if it genuinely cannot be determined
  "confidence": a float from 0.0 to 1.0 — your confidence in this specific extraction
  "source": exactly one of:
    "structured_field" — the raw data already had a clean, directly-labeled value for this field
    "free_text" — you had to infer or parse the value out of unstructured text (e.g. pulling color out of a title)
    "absent" — the field is not present anywhere in the raw data; value MUST be null when source is absent

Rules:
- category MUST be exactly one of this fixed list, verbatim — never invent a new category: {valid_categories}
- price, if present, must be a plain decimal number string with no currency symbol (e.g. "19.99")
- Never guess a value just to avoid "absent" — an absent field with null value is correct and expected when the data isn't there
- title is the one field that should essentially never be absent; if a row truly has no discernible title, extract your best available text and mark it low-confidence free_text rather than absent

Return a single JSON object with one key "rows", whose value is a JSON array.
Each array element corresponds to exactly one input row, in the same order
you were given them, and must include "row_index" matching that row's given
index, plus the {len(_FIELD_NAMES)} field objects described above.

Example shape for one row:
{{"row_index": 0, "title": {{"value": "Blue Cotton Shirt", "confidence": 0.95, "source": "free_text"}}, "brand": {{"value": "Acme", "confidence": 0.9, "source": "structured_field"}}, "category": {{"value": "shirts", "confidence": 0.85, "source": "free_text"}}, "color": {{"value": null, "confidence": 1.0, "source": "absent"}}, "material": {{"value": null, "confidence": 1.0, "source": "absent"}}, "size": {{"value": "M", "confidence": 0.9, "source": "free_text"}}, "price": {{"value": "19.99", "confidence": 0.98, "source": "structured_field"}}, "sku": {{"value": "SKU-001", "confidence": 1.0, "source": "structured_field"}}, "description": {{"value": null, "confidence": 1.0, "source": "absent"}}}}

Respond with ONLY the JSON object — no prose, no markdown fences."""


def build_user_message(rows: list[RawProduct]) -> str:
    """Per-batch, variable content. Comes AFTER the system prompt in the
    request so the stable prefix (system prompt) stays cacheable."""
    lines = [f"Extract structured fields from these {len(rows)} raw supplier rows."]
    for i, row in enumerate(rows):
        lines.append(f"Row {i}: {json.dumps(row.raw_fields, ensure_ascii=False)}")
    return "\n".join(lines)


def _decimal_or_none(value: str | None) -> Decimal | None:
    if value is None:
        return None
    try:
        return Decimal(value)
    except InvalidOperation:
        return None  # happy path: don't crash the whole batch over one unparseable price


def _to_normalized_product(raw_row: RawProduct, extracted: _RawExtractionRow) -> NormalizedProduct:
    field_provenance = {name: getattr(extracted, name) for name in _FIELD_NAMES}
    return NormalizedProduct(
        supplier_id=raw_row.supplier_id,
        source_row_index=raw_row.source_row_index,
        title=extracted.title.value or "",
        brand=extracted.brand.value,
        category=extracted.category.value or "",
        color=extracted.color.value,
        material=extracted.material.value,
        size=extracted.size.value,
        price=_decimal_or_none(extracted.price.value),
        sku=extracted.sku.value,
        description=extracted.description.value,
        field_provenance=field_provenance,
    )


def parse_extraction_response(content: str, rows: list[RawProduct]) -> list[NormalizedProduct]:
    """Parses and validates the LLM's raw JSON string, then maps each
    extracted row back to its originating RawProduct by row_index — the
    response order isn't trusted, the index is.

    Raises (deliberately, happy-path only): json.JSONDecodeError on
    malformed JSON, pydantic.ValidationError on a schema-invalid or
    out-of-taxonomy response, KeyError if row_index doesn't line up with
    the input batch. Day 5 wraps this in the reprompt-on-ValidationError
    retry loop from §1.4 — this function itself does not retry.
    """
    parsed = json.loads(content)
    batch = _RawExtractionBatch.model_validate(parsed)

    by_index = {r.row_index: r for r in batch.rows}
    missing = set(range(len(rows))) - set(by_index.keys())
    if missing:
        raise KeyError(f"LLM response missing row_index values: {sorted(missing)}")

    return [_to_normalized_product(rows[i], by_index[i]) for i in range(len(rows))]


def extract_batch(rows: list[RawProduct], client: Groq | None = None) -> list[NormalizedProduct]:
    """Runs one Groq call for the whole batch and returns one NormalizedProduct
    per input row, in the same order. `client` is injectable for testing —
    production callers can omit it and a real Groq client is built from settings.
    """
    if not rows:
        return []

    if client is None:
        client = Groq(api_key=settings.GROQ_API_KEY)

    response = client.chat.completions.create(
        model=settings.extraction_model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": build_user_message(rows)},
        ],
    )

    content = response.choices[0].message.content
    return parse_extraction_response(content, rows)
