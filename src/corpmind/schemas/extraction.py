from decimal import Decimal

from pydantic import BaseModel, Field, field_validator

from corpmind.taxonomy import load_taxonomy


class FieldExtraction(BaseModel):
    """Per-field extraction result with confidence — lets the model routing
    rule (§1.5) decide whether a field needs escalation to the bigger model."""

    value: str | None
    confidence: float = Field(ge=0.0, le=1.0)


class NormalizedProduct(BaseModel):
    """Fixed-schema product, one per RawProduct, after the Extraction agent runs."""

    supplier_id: str
    source_row_index: int
    title: str
    brand: str | None = None
    category: str
    color: str | None = None
    material: str | None = None
    size: str | None = None
    price: Decimal | None = None
    sku: str | None = None
    description: str | None = None
    field_confidences: dict[str, float] = Field(
        default_factory=dict, description="field_name -> confidence, drives escalation"
    )

    @field_validator("category")
    @classmethod
    def category_must_be_in_taxonomy(cls, v: str) -> str:
        valid = load_taxonomy()
        if v not in valid:
            raise ValueError(
                f"category '{v}' is not in the controlled taxonomy. "
                f"Valid categories: {sorted(valid)}. "
                "The LLM must pick from this list, not invent free text — "
                "this validator is the backstop when it doesn't."
            )
        return v
