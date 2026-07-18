from decimal import Decimal

from pydantic import BaseModel, Field, field_validator

from corpmind.taxonomy import load_taxonomy


class ConsistentProduct(BaseModel):
    """Final reconciled, cross supplier-merged catalog record — the output
    of the full pipeline (Matching + Enrichment + Evaluation gate passed).
    One per de-duplicated product, NOT one per raw supplier row — this is
    what the Report Agent exports, not what Extraction produces.
    """

    catalog_id: str
    sku: str | None = None
    title: str = Field(..., min_length=1)
    brand: str | None = None
    category: str
    description: str | None = None
    price: Decimal | None = Field(default=None, gt=0)
    attributes: dict[str, str] = Field(
        default_factory=dict,
        description="Extra normalized specs beyond the core fields, e.g. size/fit/pattern",
    )

    @field_validator("category")
    @classmethod
    def category_must_be_in_taxonomy(cls, v: str) -> str:
        valid = load_taxonomy()
        if v not in valid:
            raise ValueError(
                f"category '{v}' is not in the controlled taxonomy. "
                f"Valid categories: {sorted(valid)}."
            )
        return v
