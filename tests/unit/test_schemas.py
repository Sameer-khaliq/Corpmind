from decimal import Decimal

import pytest
from pydantic import ValidationError

from src.corpmind.schemas.audit import AuditLogEntry
from corpmind.schemas.consistent import ConsistentProduct
from corpmind.schemas.enrichment import EnrichmentResolution, EnrichmentResult, FieldEnrichment
from corpmind.schemas.evaluation import EvaluationRecord, FieldEvalScore, MatchEvalScore
from corpmind.schemas.extraction import FieldExtraction, NormalizedProduct
from corpmind.schemas.matching import MatchDecision, MatchResult
from corpmind.schemas.raw import RawProduct


def test_invalid_category_rejected_by_taxonomy_backstop():
    with pytest.raises(ValidationError, match="not in the controlled taxonomy"):
        NormalizedProduct(
            supplier_id="A", source_row_index=0, title="Blue Shirt",
            category="not-a-real-category",
        )


def test_valid_category_from_taxonomy_accepted():
    p = NormalizedProduct(
        supplier_id="A", source_row_index=0, title="Blue Shirt", category="shirts",
    )
    assert p.category == "shirts"


def test_matched_existing_requires_catalog_id():
    with pytest.raises(ValidationError, match="matched_catalog_id"):
        MatchResult(
            candidate_supplier_id="B", candidate_source_row_index=3,
            rrf_score=0.82, decision=MatchDecision.MATCHED_EXISTING,
            matched_catalog_id=None,
        )


def test_new_product_cannot_carry_a_catalog_id():
    with pytest.raises(ValidationError, match="only MATCHED_EXISTING"):
        MatchResult(
            candidate_supplier_id="B", candidate_source_row_index=3,
            rrf_score=0.1, decision=MatchDecision.NEW_PRODUCT,
            matched_catalog_id="cat_001",
        )


def test_filled_grounded_requires_source_url():
    with pytest.raises(ValidationError, match="source_url is missing"):
        FieldEnrichment(
            field_name="material", original_value=None, enriched_value="cotton",
            resolution=EnrichmentResolution.FILLED_GROUNDED, source_url=None,
        )


def test_left_flagged_does_not_require_source_url():
    fe = FieldEnrichment(
        field_name="material", original_value=None, enriched_value=None,
        resolution=EnrichmentResolution.LEFT_FLAGGED,
    )
    assert fe.source_url is None


def test_overall_verdict_is_review_if_any_subscore_fails():
    verdict = EvaluationRecord.derive_overall_verdict(
        match_score=MatchEvalScore(catalog_id="c1", rrf_score=0.9, verdict="accept"),
        field_scores=[
            FieldEvalScore(catalog_id="c1", field_name="material", faithfulness_score=0.6, verdict="review"),
        ],
    )
    assert verdict == "review"


def test_overall_verdict_is_accept_if_everything_passes():
    verdict = EvaluationRecord.derive_overall_verdict(
        match_score=MatchEvalScore(catalog_id="c1", rrf_score=0.9, verdict="accept"),
        field_scores=[
            FieldEvalScore(catalog_id="c1", field_name="material", faithfulness_score=0.95, verdict="accept"),
        ],
    )
    assert verdict == "accept"


# --- Day 2 checkpoint, taken literally: every model round-trips through
# model_validate(model_dump()), and a few deliberately invalid payloads
# (missing required field, out-of-range confidence, out-of-taxonomy
# category) raise ValidationError. ------------------------------------

ROUND_TRIP_CASES = [
    (RawProduct, dict(
        supplier_id="A", source_row_index=0, source_file="a.csv",
        raw_fields={"title": "Blue Shirt", "brand": None},
        warnings=["row missing values in columns: ['brand']"],
    )),
    (FieldExtraction, dict(value="Blue Shirt", confidence=0.92)),
    (NormalizedProduct, dict(
        supplier_id="A", source_row_index=0, title="Blue Shirt", category="shirts",
        price=Decimal("19.99"), field_confidences={"title": 0.92},
    )),
    (MatchResult, dict(
        candidate_supplier_id="B", candidate_source_row_index=3,
        rrf_score=0.82, decision=MatchDecision.MATCHED_EXISTING, matched_catalog_id="cat_001",
    )),
    (FieldEnrichment, dict(
        field_name="material", original_value=None, enriched_value="cotton",
        resolution=EnrichmentResolution.FILLED_GROUNDED, source_url="https://example.com",
        faithfulness_score=0.9,
    )),
    (EnrichmentResult, dict(catalog_id="cat_001", field_results=[])),
    (MatchEvalScore, dict(catalog_id="cat_001", rrf_score=0.82, verdict="accept")),
    (FieldEvalScore, dict(catalog_id="cat_001", field_name="material", faithfulness_score=0.9, verdict="accept")),
    (EvaluationRecord, dict(catalog_id="cat_001", overall_verdict="accept")),
    (AuditLogEntry, dict(
        catalog_id="cat_001", agent="matching_agent", action="merged_into_existing",
        reason="rrf_score above high cutoff",
    )),
    (ConsistentProduct, dict(
        catalog_id="cat_001", sku="TSH-01", title="Blue Cotton Shirt", category="shirts",
        price=Decimal("19.99"), attributes={"size": "M"},
    )),
]


@pytest.mark.parametrize(
    "model_cls,kwargs", ROUND_TRIP_CASES, ids=[c[0].__name__ for c in ROUND_TRIP_CASES]
)
def test_schema_round_trips_through_dump_and_validate(model_cls, kwargs):
    """This is the exact shape every one of these objects takes crossing a
    LangGraph node boundary or hitting a JSON audit log — a round-trip
    break here is a break everywhere downstream."""
    instance = model_cls(**kwargs)
    rebuilt = model_cls.model_validate(instance.model_dump())
    assert rebuilt == instance


def test_missing_required_field_raises():
    with pytest.raises(ValidationError):
        RawProduct(supplier_id="A", source_row_index=0)  # source_file, raw_fields missing


def test_out_of_range_confidence_raises():
    with pytest.raises(ValidationError):
        FieldExtraction(value="Blue Shirt", confidence=1.5)  # confidence must be in [0, 1]


def test_negative_confidence_raises():
    with pytest.raises(ValidationError):
        FieldExtraction(value="Blue Shirt", confidence=-0.1)


def test_consistent_product_rejects_generic_placeholder_category():
    """Guards against the exact regression that happened once already —
    a generic/placeholder taxonomy (Electronics, Watches, etc.) sneaking
    back in instead of the real fashion-category gold-set taxonomy."""
    with pytest.raises(ValidationError, match="not in the controlled taxonomy"):
        ConsistentProduct(
            catalog_id="cat_001", title="Blue Shirt", category="Electronics",
        )
