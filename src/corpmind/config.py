"""Central config for CorpMind. One Settings object — no scattered os.getenv() calls.

Model names, thresholds, and paths all live here so a routing change or a
threshold tweak is a one-line edit, not a grep across the codebase.
"""
import os
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Runtime environment ---------------------------------------------
    ENVIRONMENT: Literal["development", "production", "test"] = "development"
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # --- API keys -----------------------------------------------------
    GROQ_API_KEY: str
    GOOGLE_API_KEY: str  # NOT GEMINI_API_KEY — wrong name has bitten us before
    TAVILY_API_KEY: str

    # --- Storage ----------------------------------------------------------
    VECTOR_STORE_PATH: str = "./data/chroma_store"
    VECTOR_STORE_COLLECTION: str = "catalog_products"
    MATCH_HIGH_CUTOFF: float = 0.020
    MATCH_LOW_CUTOFF: float = 0.008
    # --- Model routing (per role, not hardcoded at call sites) ---------
    # Deliberately spread across 4 distinct physical Groq models so no two
    # roles share a rate-limit bucket (see rate_limiter.py's module
    # docstring on why that matters) — this is what actually cuts latency,
    # not just picking "a fast model". gemini-2.5-flash dropped from
    # judge_model: its free-tier RPM=15 was the tightest ceiling of any
    # model here and caused avoidable 429/backoff latency; kept in
    # rate_limits.yaml in case a role needs it again later, but nothing
    # currently routes to it.
    extraction_model: str = "llama-3.1-8b-instant"
    embeddings_model: str = "gemini-embedding-001"
    escalation_model: str = "llama-3.3-70b-versatile"
    judge_model: str = "openai/gpt-oss-120b"
    judge_fallback_model: str = "openai/gpt-oss-20b"
    # Enrichment's ReAct loop shares judge_fallback_model's bucket
    # (openai/gpt-oss-20b) rather than extraction_model's — extraction runs
    # once per row and would otherwise contend with enrichment's per-item
    # search loop for the same bucket, adding avoidable latency. Sharing
    # with judge_fallback is low-risk since that path only fires when the
    # primary judge errors. Revisit once nodes.py confirms how often
    # escalation_model actually gets called — if it's low-traffic too, it
    # may be a better pairing.
    enrichment_model: str = "openai/gpt-oss-20b"
    ENRICHMENT_CONFIDENCE_THRESHOLD: float = 0.6
    FAITHFULNESS_THRESHOLD: float = 0.85
    max_concurrent_llm_calls: int = 5
    # --- Thresholds (§1 of the implementation plan) --------------------
    faithfulness_threshold: float = 0.85
    match_confidence_high: float = Field(
        default=0.75, description="RRF score above this -> MATCHED_EXISTING"
    )
    match_confidence_low: float = Field(
        default=0.45, description="RRF score below this -> NEW_PRODUCT; between -> AMBIGUOUS"
    )
    disambiguation_confidence_threshold: float = Field(
        default=0.75, description="Confidence threshold for disambiguation"
    )
    # --- Config file paths ---------------------------------------------
    taxonomy_path: Path = Path("config/taxonomy.yaml")
    rate_limits_path: Path = Path("config/rate_limits.yaml")

    # --- Budget ceilings -------------------------------------------------
    tavily_monthly_credit_ceiling: int = 1000

    # --- NFR target (validated for real on Day 17) ----------------------
    batch_size_target: int = 500
    batch_time_budget_minutes: int = 15

    @field_validator("GROQ_API_KEY", "GOOGLE_API_KEY", "TAVILY_API_KEY")
    @classmethod
    def check_trailing_whitespace(cls, v: str) -> str:
        if v != v.strip():
            raise ValueError(
                "API key contains leading or trailing whitespace! Please clean your .env file."
            )
        return v

    @field_validator("GOOGLE_API_KEY")
    @classmethod
    def verify_correct_google_naming(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError(
                "GOOGLE_API_KEY is missing or empty in .env. "
                "Must be named GOOGLE_API_KEY, NOT GEMINI_API_KEY — "
                "wrong name has silently broken this before."
            )
        if "GEMINI_API_KEY" in os.environ:
            raise ValueError(
                "Detected GEMINI_API_KEY in the environment! You MUST use "
                "GOOGLE_API_KEY for the Google/Gemini client integration."
            )
        return v

    @model_validator(mode="after")
    def thresholds_must_make_sense(self) -> "Settings":
        if not (0.0 <= self.faithfulness_threshold <= 1.0):
            raise ValueError(
                f"faithfulness_threshold must be in [0,1], got {self.faithfulness_threshold}"
            )
        if self.match_confidence_low >= self.match_confidence_high:
            raise ValueError(
                f"match_confidence_low ({self.match_confidence_low}) must be < "
                f"match_confidence_high ({self.match_confidence_high}) — "
                "otherwise nothing ever lands in the AMBIGUOUS band."
            )
        return self


def check_env_file_whitespace(env_path: Path = Path(".env")) -> None:
    """Fail loud on trailing whitespace in the raw .env file text itself —
    checked before dotenv parsing even runs, as a defense-in-depth backstop
    to the field_validator above (which only sees whitespace that survives
    dotenv's own auto-stripping unquoted values).
    """
    if not env_path.exists():
        return
    for line_num, line in enumerate(env_path.read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip() and line != line.rstrip() and "=" in line and '"' not in line:
            raise ValueError(
                f"{env_path} line {line_num} has trailing whitespace: {line!r}. "
                "Fix this before continuing."
            )


check_env_file_whitespace()
settings = Settings()