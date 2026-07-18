from enum import Enum

from pydantic import BaseModel, Field, model_validator


class EnrichmentResolution(str, Enum):
    FILLED_GROUNDED = "filled_grounded"  # web-search-grounded value, faithfulness-checked
    LEFT_FLAGGED = "left_flagged"  # no reliable source found -> human review, never guessed
    NO_ACTION_NEEDED = "no_action_needed"  # field was already present, nothing to do


class FieldEnrichment(BaseModel):
    field_name: str
    original_value: str | None
    enriched_value: str | None = None
    resolution: EnrichmentResolution
    source_url: str | None = Field(
        default=None, description="Grounding source — required when resolution == filled_grounded"
    )
    faithfulness_score: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def grounded_claims_need_a_source(self) -> "FieldEnrichment":
        if self.resolution == EnrichmentResolution.FILLED_GROUNDED and not self.source_url:
            raise ValueError(
                "resolution is filled grounded but source_url is missing — "
                "nothing gets published without a traceable grounding source"
            )
        return self


class EnrichmentResult(BaseModel):
    catalog_id: str
    field_results: list[FieldEnrichment] = Field(default_factory=list)
