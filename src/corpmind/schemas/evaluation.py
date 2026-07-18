from typing import Literal

from pydantic import BaseModel, Field

Verdict = Literal["accept", "review"]


class MatchEvalScore(BaseModel):
    catalog_id: str
    rrf_score: float
    verdict: Verdict


class FieldEvalScore(BaseModel):
    catalog_id: str
    field_name: str
    faithfulness_score: float
    verdict: Verdict


class EvaluationRecord(BaseModel):
    """One per catalog_id. overall_verdict is what the graph's conditional
    edge reads to route to Report vs Human Review — no LLM in that decision,
    it's a plain Python check against this field."""

    catalog_id: str
    match_score: MatchEvalScore | None = None
    field_scores: list[FieldEvalScore] = Field(default_factory=list)
    overall_verdict: Verdict

    @classmethod
    def derive_overall_verdict(
        cls, match_score: MatchEvalScore | None, field_scores: list[FieldEvalScore]
    ) -> Verdict:
        """overall verdict is 'accept' only if every sub-score is 'accept' —
        one flagged field or a shaky match sends the whole item to review."""
        all_verdicts = [s.verdict for s in field_scores]
        if match_score is not None:
            all_verdicts.append(match_score.verdict)
        return "accept" if all_verdicts and all(v == "accept" for v in all_verdicts) else "review"
