from datetime import datetime, timezone

from pydantic import BaseModel, Field


class AuditLogEntry(BaseModel):
    """Cross-cutting — every node that makes a decision writes one of these,
    and each write should also go through logging_config's JSON logger so
    it's greppable at runtime, not just sitting in the final export."""

    catalog_id: str
    agent: str = Field(..., description="Which node/agent made this decision, e.g. 'matching_agent'")
    action: str = Field(..., description="e.g. 'merged_into_existing', 'flagged_for_review'")
    reason: str
    audit_tag: str | None = Field(
        default=None, description="e.g. 'security' for a rejected/suspected prompt-injection case"
    )
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
