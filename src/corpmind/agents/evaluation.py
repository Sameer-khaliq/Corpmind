"""
agents/evaluation_agent.py

CorpMind — Evaluation Agent
============================

Day 12 (field-eval half): wraps eval/ragas_harness.py's blind judge to score
every FILLED_GROUNDED enrichment field; LEFT_FLAGGED / NO_ACTION_NEEDED
fields pass through as trivial ACCEPT (no grounded claim was made, so there
is nothing for the judge to falsify).

Day 13 (match-eval half, same file — kept as a unit with Day 12 per the
plan's stopping-point note): MatchEvalScore from the RRF fusion score itself
(calibrated against a gold set — NOT RAGAS machinery, per §1.1's deviation),
LLM disambiguation reserved only for AMBIGUOUS matches, and overall_verdict
aggregation: ACCEPT iff match_eval AND every field_eval is ACCEPT.

Depends on eval/ragas_harness.py (Verdict, FieldFaithfulnessInput,
FieldEvalScore, evaluate_field_faithfulness_batch). ragas_harness.py has no
dependency back on this file — one-directional, no circularity.

WIRING YOU MUST DO before this runs for real:
  1. Blind-judge client — wired in ragas_harness.py's default_judge_call_fn,
     not here. Pass your real judge_call_fn into evaluate_item()/
     evaluate_enrichment_result() once that's done.
  2. `disambiguation_fn` passed into evaluate_match()/evaluate_item() — plug
     in your real llama-3.3-70b-versatile call per §1.5's routing rule. It
     must return {"resolved": bool, "confidence": float, "reasoning": str}.
  3. Config attribute names (`match_rrf_low_cutoff`, `match_rrf_high_cutoff`,
     `disambiguation_confidence_threshold`) are guessed to match the
     documented config.py pattern — confirm real names, fix the getattr()
     fallback in `_disambiguation_confidence_threshold()` if different.
  4. The `except ModuleNotFoundError` block below mirrors what schemas/
     matching.py and schemas/enrichment.py are documented to contain, so
     this file is self-testable in isolation. Once you confirm your real
     schemas match this shape, delete the except block and keep only the
     `try` import.

Run: uv run python agents/evaluation_agent.py
"""

from __future__ import annotations

from enum import Enum
from typing import Callable

from pydantic import BaseModel, Field, model_validator

try:
    from corpmind.eval.ragas_harness import (  # type: ignore
        FaithfulnessJudgeFn,
        FieldEvalScore,
        FieldFaithfulnessInput,
        Verdict,
        default_judge_call_fn,
        evaluate_field_faithfulness_batch,
    )
except ModuleNotFoundError:
    # Sandbox fallback so this file is runnable standalone next to
    # eval/ragas_harness.py without the corpmind package installed.
    from ragas_harness import (  # type: ignore
        FaithfulnessJudgeFn,
        FieldEvalScore,
        FieldFaithfulnessInput,
        Verdict,
        default_judge_call_fn,
        evaluate_field_faithfulness_batch,
    )

try:
    from corpmind.config import settings  # type: ignore
    from corpmind.logging_config import get_logger  # type: ignore
    from corpmind.schemas.enrichment import (  # type: ignore
        EnrichmentResolution,
        EnrichmentResult,
        EnrichmentSource,
        FieldEnrichment,
    )
    from corpmind.schemas.matching import MatchDecision, MatchResult  # type: ignore

    logger = get_logger(__name__)
    _REAL_IMPORTS = True
except ModuleNotFoundError:
    _REAL_IMPORTS = False
    import logging

    logger = logging.getLogger(__name__)

    class MatchDecision(str, Enum):
        NEW_PRODUCT = "NEW_PRODUCT"
        MATCHED_EXISTING = "MATCHED_EXISTING"
        AMBIGUOUS = "AMBIGUOUS"

    class EnrichmentResolution(str, Enum):
        FILLED_GROUNDED = "filled_grounded"
        LEFT_FLAGGED = "left_flagged"
        NO_ACTION_NEEDED = "no_action_needed"

    class MatchResult(BaseModel):
        catalog_id: str
        rrf_score: float
        decision: MatchDecision

    class EnrichmentSource(BaseModel):
        url: str | None = None
        snippet: str = ""

    class FieldEnrichment(BaseModel):
        field_name: str
        original_value: str | None = None
        enriched_value: str | None = None
        resolution: EnrichmentResolution
        source_url: str | None = None
        source: EnrichmentSource | None = None
        faithfulness_score: float | None = None

    class EnrichmentResult(BaseModel):
        catalog_id: str
        field_results: list[FieldEnrichment] = Field(default_factory=list)

    class _StubSettings:
        match_rrf_low_cutoff = 0.35
        match_rrf_high_cutoff = 0.65
        disambiguation_confidence_threshold = 0.75

    settings = _StubSettings()


def _disambiguation_confidence_threshold() -> float:
    return float(getattr(settings, "disambiguation_confidence_threshold", 0.75))


# ---------------------------------------------------------------------------
# Schemas — MatchEvalScore / EvaluationRecord
# (mirrors schemas/evaluation.py naming from Day 2 — reconcile if it already
# has these exact fields; add if it doesn't. FieldEvalScore lives in
# ragas_harness.py, imported above.)
# ---------------------------------------------------------------------------


class MatchEvalScore(BaseModel):
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


# ---------------------------------------------------------------------------
# Day 12 (field-eval half) — wraps ragas_harness for real EnrichmentResults
# ---------------------------------------------------------------------------


def _trivial_field_eval(fe: FieldEnrichment) -> FieldEvalScore:
    """LEFT_FLAGGED / NO_ACTION_NEEDED fields made no grounded claim — nothing
    for the judge to falsify, so they pass through as ACCEPT without a call."""
    return FieldEvalScore(
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
                    field_name=fe.field_name,
                    claimed_value=fe.enriched_value or "",
                    retrieved_snippet=(fe.source.snippet if fe.source else ""),
                )
            )
            to_judge_positions.append(i)
        else:
            output[i] = _trivial_field_eval(fe)

    judged = evaluate_field_faithfulness_batch(to_judge_inputs, judge_call_fn, batch_size, threshold)
    for pos, score in zip(to_judge_positions, judged):
        output[pos] = score

    return [o for o in output if o is not None]


# ---------------------------------------------------------------------------
# Day 13 (match-eval half) — RRF confidence + disambiguation + aggregation
# ---------------------------------------------------------------------------

DisambiguationFn = Callable[[MatchResult], dict]


def rrf_to_confidence(rrf_score: float, low_cutoff: float, high_cutoff: float) -> float:
    """
    Placeholder linear calibration mapping raw RRF fusion score onto [0,1].
    Per §1.1, the REAL calibration curve must be fit against the initial
    small labeled gold set (not built yet) — swap this out once that exists.
    Kept as a stable interface so Day 14's graph wiring and Day 22's
    regression suite aren't blocked waiting on it.
    """
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
    disambiguation_fn: DisambiguationFn | None = None,
) -> MatchEvalScore:
    confidence = rrf_to_confidence(match_result.rrf_score, low_cutoff, high_cutoff)

    if match_result.decision != MatchDecision.AMBIGUOUS:
        # Already confidently routed by the two-cutoff decision in the
        # matching agent (Day 7-8) — eval half just packages the confidence,
        # it does not re-decide.
        return MatchEvalScore(
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
    """ACCEPT iff match_eval ACCEPT and every field_eval ACCEPT. This
    function is what Day 13's done-checkpoint actually tests — not the
    individual sub-scores."""
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


# ---------------------------------------------------------------------------
# Top-level orchestrator — the node LangGraph's evaluation gate calls
# (Day 14 wiring point)
# ---------------------------------------------------------------------------


def evaluate_item(
    catalog_id: str,
    match_result: MatchResult,
    enrichment_result: EnrichmentResult | None,
    low_cutoff: float,
    high_cutoff: float,
    judge_call_fn: FaithfulnessJudgeFn = default_judge_call_fn,
    disambiguation_fn: DisambiguationFn | None = None,
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


# ---------------------------------------------------------------------------
# Smoke tests — Day 13's named Done checkpoint, plus one bonus check for the
# AMBIGUOUS/disambiguation path so it isn't left silently untested.
# (Day 12's own checkpoint already lives in eval/ragas_harness.py — this file
# re-proves it end-to-end through evaluate_enrichment_result too.)
# Run: uv run python agents/evaluation_agent.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # === Day 13 checkpoint ==================================================
    # Mixed case: good match, one faithful field, one unfaithful field.
    # Tests the AGGREGATION — match_eval alone would ACCEPT, but one bad
    # field must still flip the overall verdict to REJECT_TO_REVIEW.
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
                enriched_value="navy blue",
                resolution=EnrichmentResolution.FILLED_GROUNDED,
                source_url="https://example.com/a",
                source=EnrichmentSource(url="https://example.com/a", snippet="colour: navy blue, size M-XL"),
            ),
            FieldEnrichment(
                field_name="material",
                enriched_value="100% cotton",
                resolution=EnrichmentResolution.FILLED_GROUNDED,
                source_url="https://example.com/b",
                source=EnrichmentSource(
                    url="https://example.com/b", snippet="material composition not listed on this page"
                ),
            ),
            FieldEnrichment(
                field_name="size",
                enriched_value=None,
                resolution=EnrichmentResolution.LEFT_FLAGGED,
                source_url=None,
                source=None,
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

    # === Bonus check (not a named checkpoint, but proves this path isn't dead) ==
    ambiguous_match = MatchResult(catalog_id="cat-002", rrf_score=0.5, decision=MatchDecision.AMBIGUOUS)

    def _mock_disambiguation(match_result: MatchResult) -> dict:
        return {"resolved": True, "confidence": 0.9, "reasoning": "title + brand match after LLM comparison"}

    ambiguous_eval = evaluate_match(
        ambiguous_match, low_cutoff=0.35, high_cutoff=0.65, disambiguation_fn=_mock_disambiguation
    )
    assert ambiguous_eval.verdict == "ACCEPT"
    assert ambiguous_eval.disambiguation_used is True
    print("[Bonus] PASS — AMBIGUOUS match resolved via disambiguation:", ambiguous_eval.reason)

    print("\nevaluation_agent.py checkpoints passed. (real_imports =", _REAL_IMPORTS, ")")
