from enum import Enum

from pydantic import BaseModel, Field, model_validator


class MatchDecision(str, Enum):
    NEW_PRODUCT = "NEW_PRODUCT"
    MATCHED_EXISTING = "MATCHED_EXISTING"
    AMBIGUOUS = "AMBIGUOUS"  # skips Enrichment, goes straight to human review


class MatchResult(BaseModel):
    """Output of the Matching/Dedup agent for one normalized product.

    rrf_score is the actual confidence signal (calibrated against the gold
    set) — NOT a RAGAS score. RAGAS faithfulness is reserved for Enrichment's
    grounding claims only; see schemas -enrichment.py.
    """

    candidate_supplier_id: str
    candidate_source_row_index: int
    matched_catalog_id: str | None = Field(
        default=None, description="Set only when decision == MATCHED_EXISTING"
    )
    rrf_score: float
    decision: MatchDecision

    @model_validator(mode="after")
    def matched_id_consistency(self) -> "MatchResult":
        if self.decision == MatchDecision.MATCHED_EXISTING and self.matched_catalog_id is None:
            raise ValueError("decision is MATCHED_EXISTING but matched_catalog_id is not set")
        if self.decision != MatchDecision.MATCHED_EXISTING and self.matched_catalog_id is not None:
            raise ValueError(
                f"decision is {self.decision} but matched_catalog_id is set — "
                "only MATCHED_EXISTING should carry a target id"
            )
        return self
