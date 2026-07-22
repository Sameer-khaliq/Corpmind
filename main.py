
from __future__ import annotations
from enum import Enum
from typing import Callable
from pydantic import BaseModel, Field, model_validator
import logging
# Strict internal package imports
from corpmind.eval.ragas_harness import (  
    FaithfulnessJudgeFn,
    FieldEvalScore,  
    FieldFaithfulnessInput,
    Verdict,
    default_judge_call_fn,
    evaluate_field_faithfulness_batch,
)
from corpmind.config import settings  
from corpmind.schemas.enrichment import (  
    EnrichmentResolution,
    EnrichmentResult,
    EnrichmentSource,
    FieldEnrichment,
)
from corpmind.schemas.matching import MatchDecision, MatchResult

logger = logging.getLogger(__name__)
_REAL_IMPORTS = True
print("Running main.py as a script. This is for testing and debugging purposes only.")

