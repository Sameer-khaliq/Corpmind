from enum import Enum

from pydantic import BaseModel, Field, model_validator


class MatchDecision(str, Enum):
    NEW_PRODUCT = "NEW_PRODUCT"
    MATCHED_EXISTING = "MATCHED_EXISTING"
    AMBIGUOUS = "AMBIGUOUS"


class MatchResult(BaseModel):
    """Output of the Matching/Dedup agent for one normalized product.

    rrf_score is the actual confidence signal (calibrated against the gold
    set) — NOT a RAGAS score. RAGAS faithfulness is reserved for Enrichment's
    grounding claims only; see schemas/enrichment.py.
    """

    catalog_id: str | None = Field(
        default=None,
        description="Set for MATCHED_EXISTING (reused id) and NEW_PRODUCT "
        "(freshly minted id, shared across an intra-batch duplicate cluster). "
        "Left None only for AMBIGUOUS, where no id has been assigned yet.",
    )
    rrf_score: float
    decision: MatchDecision

    @model_validator(mode="after")
    def catalog_id_consistency(self) -> "MatchResult":
        needs_id = self.decision in (MatchDecision.MATCHED_EXISTING, MatchDecision.NEW_PRODUCT)
        if needs_id and self.catalog_id is None:
            raise ValueError(f"decision is {self.decision} but catalog_id is not set")
        if not needs_id and self.catalog_id is not None:
            raise ValueError(
                f"decision is {self.decision} but catalog_id is set — "
                "AMBIGUOUS should not carry a target id"
            )
        return self