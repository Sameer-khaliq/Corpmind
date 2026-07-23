from __future__ import annotations
from enum import Enum
from typing import Callable
from pydantic import BaseModel, Field, model_validator
import logging
# Strict internal package imports
from src.corpmind.eval.ragas_harness import (  
    FaithfulnessJudgeFn,
    FieldEvalScore,  
    FieldFaithfulnessInput,
    Verdict,
    default_judge_call_fn,
    evaluate_field_faithfulness_batch,
)
from src.corpmind.config import settings  
from src.corpmind.schemas.enrichment import (  
    EnrichmentResolution,
    EnrichmentResult,
    EnrichmentSource,
    FieldEnrichment,
)
from src.corpmind.schemas.matching import MatchDecision, MatchResult

logger = logging.getLogger(__name__)
_REAL_IMPORTS = True

def _disambiguation_confidence_threshold() -> float:
    return float(getattr(settings, "disambiguation_confidence_threshold", 0.75))


class MatchEvalScore(BaseModel):
    catalog_id: str | None = None
    rrf_score: float
    decision: MatchDecision
    confidence: float = Field(ge=0.0, le=1.0)
    verdict: Verdict
    reason: str
    disambiguation_used: bool = False


class EvaluationRecord(BaseModel):
    catalog_id: str
    match_eval: MatchEvalScore
    field_evals: list[FieldEvalScore] = Field(default_factory=list)
    overall_verdict: Verdict
    overall_reason: str

    @model_validator(mode="after")
    def _verdict_matches_subscores(self) -> "EvaluationRecord":
        expected, _ = aggregate_verdict(self.match_eval, self.field_evals)
        if self.overall_verdict != expected:
            raise ValueError(
                f"overall_verdict={self.overall_verdict!r} is inconsistent with "
                f"sub-scores (match_eval={self.match_eval.verdict}, "
                f"field_evals={[fe.verdict for fe in self.field_evals]}) — "
                f"expected {expected!r}."
            )
        return self

def _trivial_field_eval(catalog_id: str, fe: FieldEnrichment) -> FieldEvalScore:
    """LEFT_FLAGGED / NO_ACTION_NEEDED fields made no grounded claim — nothing
    for the judge to falsify, so they pass through as ACCEPT without a call."""
    return FieldEvalScore(
        catalog_id=catalog_id,
        field_name=fe.field_name,
        claimed_value=fe.enriched_value or "",
        faithfulness_score=1.0,
        verdict="ACCEPT",
        reason=(
            f"ACCEPT — resolution={fe.resolution}; no grounded claim was made "
            "for this field, so there is nothing for the faithfulness judge to falsify."
        ),
        injection_suspected=False,
    )


def evaluate_enrichment_result(
    result: EnrichmentResult,
    judge_call_fn: FaithfulnessJudgeFn = default_judge_call_fn,
    batch_size: int = 8,
    threshold: float | None = None,
) -> list[FieldEvalScore]:
    """Splits FILLED_GROUNDED (needs judge, via ragas_harness) from
    LEFT_FLAGGED/NO_ACTION_NEEDED (trivial ACCEPT), reassembles in original
    field order."""
    to_judge_inputs: list[FieldFaithfulnessInput] = []
    to_judge_positions: list[int] = []
    output: list[FieldEvalScore | None] = [None] * len(result.field_results)

    for i, fe in enumerate(result.field_results):
        if fe.resolution == EnrichmentResolution.FILLED_GROUNDED:
            to_judge_inputs.append(
                FieldFaithfulnessInput(
                    catalog_id=result.catalog_id,
                    field_name=fe.field_name,
                    claimed_value=fe.enriched_value or "",
                    retrieved_snippet=(fe.source.snippet if fe.source else ""),
                )
            )
            to_judge_positions.append(i)
        else:
            output[i] = _trivial_field_eval(result.catalog_id, fe)

    judged = evaluate_field_faithfulness_batch(to_judge_inputs, judge_call_fn, batch_size, threshold)
    for pos, score in zip(to_judge_positions, judged):
        output[pos] = score

    return [o for o in output if o is not None]

DisambiguationFn = Callable[[MatchResult], dict]


def default_disambiguation_fn(match_result: MatchResult) -> dict:
    """
    Real implementation — calls Groq's llama-3.3-70b-versatile to resolve an
    AMBIGUOUS match, per §1.5's model routing rule.
 
    FLAGGED PLAINLY: MatchResult here only carries catalog_id / rrf_score /
    decision — no actual item-vs-candidate product fields (title, brand,
    category, etc.). Without those an LLM has nothing real to compare, and
    this call degrades to guessing off a bare number. The getattr() calls
    below fall back gracefully so this won't crash if those fields are
    missing, but it also won't be doing genuine disambiguation until your
    real corpmind.schemas.matching.MatchResult actually carries that data
    (or it's threaded through some other way). Confirm this before trusting
    the AMBIGUOUS path in production.
    """
    import json
 
    from groq import Groq
 
    client = Groq(api_key=settings.GROQ_API_KEY)
 
    item_desc = getattr(match_result, "item_summary", None) or getattr(match_result, "normalized_product", None)
    candidate_desc = getattr(match_result, "candidate_summary", None) or getattr(match_result, "matched_candidate", None)
 
    prompt = (
        "You are resolving an AMBIGUOUS product match in a catalog "
        "reconciliation pipeline. The hybrid-search RRF fusion score for "
        f"this pair was {match_result.rrf_score:.4f}, which fell between the "
        "two decision cutoffs (too high to call a new product, too low to "
        "call a confident duplicate).\n\n"
        f"Item being matched:\n{item_desc if item_desc else '(no item detail available — rrf_score only)'}\n\n"
        f"Candidate existing catalog entry (catalog_id={match_result.catalog_id}):\n"
        f"{candidate_desc if candidate_desc else '(no candidate detail available — rrf_score only)'}\n\n"
        "Decide whether these are the SAME real-world product (described "
        "differently across suppliers) or DIFFERENT products. Precision "
        "matters more than recall — merging two different products is worse "
        "than missing a duplicate.\n\n"
        'Return ONLY a JSON object: {"resolved": <bool>, "confidence": '
        '<float 0.0-1.0>, "reasoning": "<one or two sentences>"}\n'
        '"resolved": true only if you are genuinely confident either way. If '
        "the available detail is insufficient to judge, return "
        "resolved=false with low confidence rather than guessing."
    )
 
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
 
    raw_text = response.choices[0].message.content
 
    try:
        parsed = json.loads(raw_text)
        return {
            "resolved": bool(parsed.get("resolved", False)),
            "confidence": float(parsed.get("confidence", 0.0)),
            "reasoning": str(parsed.get("reasoning", "no reasoning returned")),
        }
    except Exception as e:
        logger.warning("disambiguation response could not be parsed for %s: %s", match_result.catalog_id, e)
        return {"resolved": False, "confidence": 0.0, "reasoning": f"unparseable disambiguation response: {raw_text[:200]}"}


def rrf_to_confidence(rrf_score: float, low_cutoff: float, high_cutoff: float) -> float:
    if high_cutoff <= low_cutoff:
        raise ValueError("high_cutoff must be greater than low_cutoff")
    if rrf_score <= low_cutoff:
        return 0.0
    if rrf_score >= high_cutoff:
        return 1.0
    return (rrf_score - low_cutoff) / (high_cutoff - low_cutoff)


def _run_disambiguation(match_result: MatchResult, disambiguation_fn: DisambiguationFn | None) -> dict:
    if disambiguation_fn is None:
        raise NotImplementedError(
            "AMBIGUOUS match reached evaluate_match without a disambiguation_fn "
            "wired in. Wire this to llama-3.3-70b-versatile per §1.5's model "
            "routing rule — it must return {'resolved': bool, 'confidence': "
            "float, 'reasoning': str}."
        )
    return disambiguation_fn(match_result)


def evaluate_match(
    match_result: MatchResult,
    low_cutoff: float,
    high_cutoff: float,
    disambiguation_fn: DisambiguationFn = default_disambiguation_fn,
) -> MatchEvalScore:
    confidence = rrf_to_confidence(match_result.rrf_score, low_cutoff, high_cutoff)

    if match_result.decision != MatchDecision.AMBIGUOUS:
        return MatchEvalScore(
            catalog_id=match_result.catalog_id,
            rrf_score=match_result.rrf_score,
            decision=match_result.decision,
            confidence=confidence,
            verdict="ACCEPT",
            reason=(
                f"ACCEPT — decision={match_result.decision} from RRF "
                f"{match_result.rrf_score:.4f} (confidence {confidence:.2f}); "
                f"outside the ambiguous band [{low_cutoff}, {high_cutoff}]."
            ),
            disambiguation_used=False,
        )

    disambig = _run_disambiguation(match_result, disambiguation_fn)
    resolved = bool(disambig.get("resolved", False))
    disambig_confidence = float(disambig.get("confidence", 0.0))
    reasoning = disambig.get("reasoning", "no reasoning returned")

    if resolved and disambig_confidence >= _disambiguation_confidence_threshold():
        return MatchEvalScore(
            catalog_id=match_result.catalog_id,
            rrf_score=match_result.rrf_score,
            decision=match_result.decision,
            confidence=disambig_confidence,
            verdict="ACCEPT",
            reason=(
                f"ACCEPT — AMBIGUOUS match resolved by LLM disambiguation "
                f"({disambig_confidence:.2f} confidence): {reasoning}"
            ),
            disambiguation_used=True,
        )

    return MatchEvalScore(
        catalog_id=match_result.catalog_id,
        rrf_score=match_result.rrf_score,
        decision=match_result.decision,
        confidence=disambig_confidence,
        verdict="REJECT_TO_REVIEW",
        reason=(
            "REJECT_TO_REVIEW — AMBIGUOUS match, LLM disambiguation did not "
            f"resolve confidently: {reasoning}"
        ),
        disambiguation_used=True,
    )


def aggregate_verdict(match_eval: MatchEvalScore, field_evals: list[FieldEvalScore]) -> tuple[Verdict, str]:
    failing_fields = [fe for fe in field_evals if fe.verdict != "ACCEPT"]
    if match_eval.verdict == "ACCEPT" and not failing_fields:
        return "ACCEPT", "ACCEPT — match_eval and all field_evals passed."

    reasons = []
    if match_eval.verdict != "ACCEPT":
        reasons.append(f"match_eval rejected ({match_eval.reason})")
    if failing_fields:
        names = ", ".join(fe.field_name for fe in failing_fields)
        reasons.append(f"field_eval(s) rejected for: {names}")
    return "REJECT_TO_REVIEW", "REJECT_TO_REVIEW — " + "; ".join(reasons)

def evaluate_item(
    catalog_id: str,
    match_result: MatchResult,
    enrichment_result: EnrichmentResult | None,
    low_cutoff: float,
    high_cutoff: float,
    judge_call_fn: FaithfulnessJudgeFn = default_judge_call_fn,
    disambiguation_fn: DisambiguationFn = default_disambiguation_fn,
    batch_size: int = 8,
    threshold: float | None = None,
) -> EvaluationRecord:
    match_eval = evaluate_match(match_result, low_cutoff, high_cutoff, disambiguation_fn)

    field_evals: list[FieldEvalScore] = []
    if enrichment_result is not None:
        field_evals = evaluate_enrichment_result(enrichment_result, judge_call_fn, batch_size, threshold)

    overall_verdict, overall_reason = aggregate_verdict(match_eval, field_evals)

    return EvaluationRecord(
        catalog_id=catalog_id,
        match_eval=match_eval,
        field_evals=field_evals,
        overall_verdict=overall_verdict,
        overall_reason=overall_reason,
    )
if __name__ == "__main__":

    good_match = MatchResult(catalog_id="cat-001", rrf_score=0.9, decision=MatchDecision.MATCHED_EXISTING)

    def _mock_judge(batch: list[FieldFaithfulnessInput]) -> list[dict]:
        out = []
        for pair in batch:
            if pair.field_name == "color":
                out.append({"score": 0.95, "directive": "SUPPORTED", "evidence_span": "colour: navy blue"})
            else:
                out.append({"score": 0.10, "directive": "NOT_SUPPORTED", "evidence_span": ""})
        return out

    enrichment = EnrichmentResult(
        catalog_id="cat-001",
        field_results=[
            FieldEnrichment(
                field_name="color",
                original_value="None",
                enriched_value="navy blue",
                resolution=EnrichmentResolution.FILLED_GROUNDED,
                source_url="https://example.com/a",
                source=EnrichmentSource(url="https://example.com/a", snippet="colour: navy blue, size M-XL"),
                 faithfulness_score=0.95,
            ),
            FieldEnrichment(
                field_name="material",
                original_value="None",
                enriched_value="100% cotton",
                resolution=EnrichmentResolution.FILLED_GROUNDED,
                source_url="https://example.com/b",
                source=EnrichmentSource(
                    url="https://example.com/b", snippet="material composition not listed on this page"
                ),
                faithfulness_score=0.10,
            ),
            FieldEnrichment(
                field_name="size",
                original_value="None",
                enriched_value=None,
                resolution=EnrichmentResolution.LEFT_FLAGGED,
                source_url=None,
                source=None,
                faithfulness_score=None,
            ),
        ],
    )

    day13_record = evaluate_item(
        catalog_id="cat-001",
        match_result=good_match,
        enrichment_result=enrichment,
        low_cutoff=0.35,
        high_cutoff=0.65,
        judge_call_fn=_mock_judge,
    )
    assert day13_record.match_eval.verdict == "ACCEPT"
    field_verdicts = {fe.field_name: fe.verdict for fe in day13_record.field_evals}
    assert field_verdicts["color"] == "ACCEPT"
    assert field_verdicts["material"] == "REJECT_TO_REVIEW"
    assert field_verdicts["size"] == "ACCEPT"  
    assert day13_record.overall_verdict == "REJECT_TO_REVIEW", "aggregation must reject even though match_eval alone would accept"
    print("[Day 13] PASS — mixed case aggregated correctly:", day13_record.overall_reason)

  
    ambiguous_match = MatchResult(rrf_score=0.5, decision=MatchDecision.AMBIGUOUS)

    def _mock_disambiguation(match_result: MatchResult) -> dict:
        return {"resolved": True, "confidence": 0.9, "reasoning": "title + brand match after LLM comparison"}

    ambiguous_eval = evaluate_match(
        ambiguous_match, low_cutoff=0.35, high_cutoff=0.65, disambiguation_fn=_mock_disambiguation
    )
    assert ambiguous_eval.verdict == "ACCEPT"
    assert ambiguous_eval.disambiguation_used is True
    print("[Bonus] PASS — AMBIGUOUS match resolved via disambiguation:", ambiguous_eval.reason)

    print("\nevaluation_agent.py checkpoints passed. (real_imports =", _REAL_IMPORTS, ")")