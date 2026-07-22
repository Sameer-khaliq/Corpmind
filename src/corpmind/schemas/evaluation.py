from typing import Literal
from enum import Enum
from pydantic import BaseModel, Field

Verdict = Literal["accept", "review"]

class MatchDecision(str, Enum):
    NEW_PRODUCT = "NEW_PRODUCT"
    MATCHED_EXISTING = "MATCHED_EXISTING"
    AMBIGUOUS = "AMBIGUOUS"

class MatchEvalScore(BaseModel):
    catalog_id: str
    rrf_score: float
    decision: MatchDecision
    confidence: float = Field(ge=0.0, le=1.0)
    verdict: Verdict
    reason: str
    disambiguation_used: bool = False

class FieldEvalScore(BaseModel):
    catalog_id: str                      
    field_name: str
    claimed_value: str
    faithfulness_score: float = Field(ge=0.0, le=1.0)
    verdict: Verdict                     
    reason: str
    injection_suspected: bool = False

class EvaluationRecord(BaseModel):
    catalog_id: str
    match_score: MatchEvalScore | None = None          
    field_scores: list[FieldEvalScore] = Field(default_factory=list) 
    overall_verdict: Verdict
    overall_reason: str

    @classmethod
    def derive_overall_verdict(
        cls, match_score: MatchEvalScore | None, field_scores: list[FieldEvalScore]
    ) -> Verdict:
        all_verdicts = [s.verdict for s in field_scores]
        if match_score is not None:
            all_verdicts.append(match_score.verdict)
        return "accept" if all_verdicts and all(v == "accept" for v in all_verdicts) else "review"