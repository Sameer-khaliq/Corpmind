"""
orchestration/rate_limiter.py

CorpMind — Token-bucket rate limiter (Day 16)
=================================================

One token-bucket PAIR (requests + tokens) per physical MODEL — not per
role/config-key. This distinction matters: two roles can resolve to the
same underlying model (escalation_model and judge_fallback_model both
default to llama-3.3-70b-versatile in config.py), and Groq/Google meter
RPM/TPM per model, not per role. Bucketing by role would silently double
your effective on-paper budget while the provider still enforces one real
ceiling server-side — you'd only find out at 429 time. get_model_limiter()
below resolves and caches by MODEL NAME, so both roles share one bucket.

Limits are read from settings.rate_limits_path (config/rate_limits.yaml),
never hardcoded — per §1.3's Gemini caveat: Google's free-tier numbers are
no longer a static published table, so whatever goes in that YAML for
gemini-2.5-flash needs to come from YOUR account's live dashboard at
aistudio.google.com/rate-limit, checked on the day you run this for real —
not trusted from the YAML's placeholder or from me.

WIRING / VERIFICATION YOU MUST DO:
  1. config/rate_limits.yaml ships alongside this file with real Groq
     numbers and PLACEHOLDER zeros for both Gemini models. A zero-capacity
     bucket raises ValueError the instant that model is first requested
     (see TokenBucket.__init__) — loud and immediate, not a silent hang —
     but you still have to go fill in the real numbers before Gemini calls
     can go through rate_limited().
  2. This needs PyYAML. It is NOT currently in your pyproject.toml — given
     the packages-installed-without-`uv add`-silently-break-Docker history
     you already have, don't assume a transitive install covers it:
     `uv add pyyaml`.
  3. rate_limited() below is the wrapper — I don't have nodes.py's real
     call sites (the file uploaded under that name contained the §1.3 text,
     not code, so I never saw your actual node implementations). Apply the
     decorator directly above whichever function in nodes.py issues each
     real model call — see rate_limited()'s own docstring for the exact
     pattern. I can't do that wiring for you without seeing that file.
  4. This bucket is in-process and starts full on every process start. If
     you restart the batch runner shortly after a previous run against the
     same API key, the provider's real server-side window may already be
     partially consumed in a way this bucket doesn't know about — that
     residual risk is exactly what GROQ_RETRY_POLICY / GEMINI_RETRY_POLICY
     in tracing_config.py already exist to absorb (a stray 429 becomes a
     TransientAPIError retry, not a crash). The bucket minimizes 429s; the
     retry policy mops up the ones it can't prevent. Neither replaces the
     other.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import time
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field

try:
    import yaml  # type: ignore

    _HAS_YAML = True
except ModuleNotFoundError:
    _HAS_YAML = False

try:
    from corpmind.config import settings  # type: ignore

    _HAS_REAL_SETTINGS = True
except ModuleNotFoundError:
    _HAS_REAL_SETTINGS = False

    class _StubSettings:
        rate_limits_path = Path("config/rate_limits.yaml")
        max_concurrent_llm_calls = 10
        extraction_model = "llama-3.1-8b-instant"
        escalation_model = "llama-3.3-70b-versatile"
        judge_model = "gemini-2.5-flash"
        judge_fallback_model = "llama-3.3-70b-versatile"
        embeddings_model = "gemini-embeddings-001"
        tavily_monthly_credit_ceiling = 1000

    settings = _StubSettings()  # type: ignore

logger = logging.getLogger(__name__)


class RateLimitConfigError(Exception):
    """Anything wrong with rate_limits.yaml itself — missing file,
    malformed YAML, or a model referenced by settings that has no entry.
    Raised at load/lookup time, not discovered later mid-batch."""


class TavilyBudgetExceededError(Exception):
    """Raised when a spend() call would exceed the configured monthly
    Tavily credit ceiling."""


class ModelLimits(BaseModel):
    """rpm/tpm for one physical model or provider, as loaded from YAML.
    ge=0 (not gt=0) deliberately allows the 0-placeholder pattern used for
    not-yet-verified Gemini numbers — the file must still LOAD even if one
    entry isn't filled in yet. TokenBucket is what actually refuses to
    construct a 0-capacity bucket; see its __init__.
    """

    rpm: float = Field(ge=0)
    tpm: float | None = Field(default=None, ge=0)


def load_rate_limits(path: Path | None = None) -> dict[str, ModelLimits]:
    """Reads model_name -> ModelLimits from YAML (models: + providers:
    sections merged into one flat namespace). Raises RateLimitConfigError —
    never silently falls back to a guessed number — if PyYAML is missing,
    the file is missing, or an entry is malformed."""
    resolved_path = Path(path) if path is not None else Path(getattr(settings, "rate_limits_path", "config/rate_limits.yaml"))

    if not _HAS_YAML:
        raise RateLimitConfigError("PyYAML isn't installed. Run `uv add pyyaml` — it's not in pyproject.toml today.")
    if not resolved_path.exists():
        raise RateLimitConfigError(
            f"{resolved_path} not found. rate_limiter.py refuses to guess your account's real RPM/TPM "
            f"numbers — create this file (a starter ships alongside this module) and fill in your live "
            f"Gemini numbers from aistudio.google.com/rate-limit before using those models."
        )

    raw = yaml.safe_load(resolved_path.read_text(encoding="utf-8")) or {}
    combined = {**raw.get("models", {}), **raw.get("providers", {})}

    parsed: dict[str, ModelLimits] = {}
    for name, cfg in combined.items():
        if not isinstance(cfg, dict) or "rpm" not in cfg:
            raise RateLimitConfigError(f"{resolved_path}: entry '{name}' is missing required key 'rpm'")
        parsed[name] = ModelLimits(rpm=cfg["rpm"], tpm=cfg.get("tpm"))
    return parsed


class TokenBucket:
    """Continuous-refill token bucket. `capacity` tokens refill linearly
    over `refill_period_seconds`. acquire(amount) blocks (async) until
    enough tokens are available, then deducts them. This is what actually
    paces call START times — it has nothing to do with how many calls are
    in flight at once (that's max_concurrency's job, a separate mechanism;
    see batch_runner.py and §1.3's "two mechanisms, not one" point)."""

    def __init__(self, capacity: float, refill_period_seconds: float = 60.0):
        if capacity <= 0:
            raise ValueError(
                f"capacity must be positive, got {capacity} — refusing to construct a bucket that can "
                f"never grant a single acquire(). If this came from rate_limits.yaml, that model's "
                f"rpm/tpm is still an unfilled 0-placeholder; fill it in before using this model."
            )
        if refill_period_seconds <= 0:
            raise ValueError(f"refill_period_seconds must be positive, got {refill_period_seconds}")
        self.capacity = float(capacity)
        self.refill_period_seconds = float(refill_period_seconds)
        self._tokens = float(capacity)  # starts full — see WIRING note #4 above on what this does/doesn't know
        self._refill_rate = self.capacity / self.refill_period_seconds  # tokens per second
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    def _refill_locked(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        if elapsed > 0:
            self._tokens = min(self.capacity, self._tokens + elapsed * self._refill_rate)
            self._last_refill = now

    async def acquire(self, amount: float = 1.0) -> None:
        if amount > self.capacity:
            raise ValueError(
                f"requested {amount} exceeds bucket capacity {self.capacity} — this can never be "
                f"satisfied even from a full, otherwise-idle bucket"
            )
        while True:
            async with self._lock:
                self._refill_locked()
                if self._tokens >= amount:
                    self._tokens -= amount
                    return
                deficit = amount - self._tokens
                wait_seconds = deficit / self._refill_rate
            await asyncio.sleep(wait_seconds)

    @property
    def available(self) -> float:
        """Rough gauge only (reads without refilling/locking) — never used
        as the basis for an acquire decision itself."""
        return self._tokens


class ModelRateLimiter:
    """One of these per physical model/provider: a request-count bucket
    (always) plus a token-count bucket (only if the model meters TPM —
    Tavily doesn't)."""

    def __init__(self, name: str, limits: ModelLimits):
        self.name = name
        self.request_bucket = TokenBucket(capacity=limits.rpm, refill_period_seconds=60.0)
        self.token_bucket = TokenBucket(capacity=limits.tpm, refill_period_seconds=60.0) if limits.tpm else None

    async def acquire(self, estimated_tokens: float = 1.0) -> None:
        await self.request_bucket.acquire(1.0)
        if self.token_bucket is not None:
            await self.token_bucket.acquire(estimated_tokens)


# --------------------------------------------------------------------------
# Registry — cached by resolved MODEL NAME, loaded lazily from YAML once.
# --------------------------------------------------------------------------

_limiters: dict[str, ModelRateLimiter] = {}
_loaded_limits: dict[str, ModelLimits] | None = None


def _get_loaded_limits() -> dict[str, ModelLimits]:
    global _loaded_limits
    if _loaded_limits is None:
        _loaded_limits = load_rate_limits()
    return _loaded_limits


def get_model_limiter(model_name: str) -> ModelRateLimiter:
    """Cached by resolved model name (not by role) — two roles pointing at
    the same physical model share the same limiter instance; see module
    docstring. Construction is idempotent/side-effect-free, so a rare
    double-construction under first-use concurrency is harmless — not
    worth a lock for."""
    if model_name not in _limiters:
        limits_by_name = _get_loaded_limits()
        if model_name not in limits_by_name:
            raise RateLimitConfigError(
                f"No rate-limit entry for model '{model_name}' in "
                f"{getattr(settings, 'rate_limits_path', '(unknown path)')}. Add one before routing calls "
                f"to this model through rate_limited() — refusing to guess a number."
            )
        _limiters[model_name] = ModelRateLimiter(model_name, limits_by_name[model_name])
    return _limiters[model_name]


def _resolve_model_name(model_setting: str) -> str:
    """model_setting is usually a settings ATTRIBUTE NAME ("extraction_model")
    so routing changes (repointing extraction_model at a different model)
    keep working without touching the decorator. For providers with no role
    indirection (e.g. "tavily", which has no settings.tavily_model), the
    string is used as the literal model/provider name directly."""
    return getattr(settings, model_setting) if hasattr(settings, model_setting) else model_setting


def rate_limited(model_setting: str, estimate_tokens: Callable[..., float] | float = 1.0) -> Callable:
    """
    Decorator for an ASYNC function that issues ONE call to the model named
    by settings.<model_setting> (e.g. "extraction_model" ->
    "llama-3.1-8b-instant"), or, for providers with no role indirection
    (e.g. "tavily"), the literal name itself.

    estimate_tokens:
      - a flat float (rough per-call estimate), or
      - a callable(*args, **kwargs) -> float computed from the actual
        arguments the wrapped function receives, for when a flat constant
        would be too far off the real payload size to trust the TPM pacing.
        Per §1.3: measure your real per-item token cost, don't trust a
        guessed constant long-term.

    Wrap the function that calls the MODEL specifically — not the whole
    node — since a node like extract_and_phase_a may do non-LLM work
    (Phase A logic) alongside the LLM call, and gating all of it behind one
    model's bucket would serialize work that has nothing to do with that
    model's quota.

    Changes nothing about the wrapped function's signature, return value,
    or exceptions — it only awaits the bucket(s) before the call runs.
    """

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            model_name = _resolve_model_name(model_setting)
            limiter = get_model_limiter(model_name)
            tokens = estimate_tokens(*args, **kwargs) if callable(estimate_tokens) else estimate_tokens
            await limiter.acquire(tokens)
            return await fn(*args, **kwargs)

        return wrapper

    return decorator


class TavilyCreditBudget:
    """Monthly-scoped, decrementing credit budget. Tracks spend for THIS
    PROCESS's lifetime only — persisting spend across process restarts, so
    a real calendar month's ceiling holds across multiple separate runs, is
    explicitly OUT of scope here. That would need a small on-disk or
    DB-backed counter keyed by (year, month); Day 16 doesn't build one."""

    def __init__(self, ceiling: int):
        self.ceiling = ceiling
        self._spent = 0
        self._lock = asyncio.Lock()

    async def spend(self, credits: int = 1) -> None:
        async with self._lock:
            if self._spent + credits > self.ceiling:
                raise TavilyBudgetExceededError(
                    f"Spending {credits} more credit(s) would exceed the {self.ceiling}-credit monthly "
                    f"ceiling (already spent {self._spent} this process)."
                )
            self._spent += credits

    @property
    def remaining(self) -> int:
        return self.ceiling - self._spent


_tavily_budget: TavilyCreditBudget | None = None


def get_tavily_budget() -> TavilyCreditBudget:
    global _tavily_budget
    if _tavily_budget is None:
        _tavily_budget = TavilyCreditBudget(int(getattr(settings, "tavily_monthly_credit_ceiling", 1000)))
    return _tavily_budget


def assert_rate_not_exceeded(
    timestamps: list[float],
    ceiling: float,
    window_seconds: float,
    weights: list[float] | None = None,
    label: str = "rate",
) -> None:
    """Sliding-window check over recorded call-start timestamps: no window
    of length window_seconds may contain more than `ceiling` total weight
    (weights default to 1.0 each, i.e. a plain request-count check; pass
    per-call token counts as weights to check a TPM ceiling instead)."""
    if weights is not None and len(weights) != len(timestamps):
        raise ValueError("weights must be the same length as timestamps")
    events = sorted(zip(timestamps, weights or [1.0] * len(timestamps)))
    window: list[tuple[float, float]] = []
    running_sum = 0.0
    for t, w in events:
        window.append((t, w))
        running_sum += w
        while window[0][0] <= t - window_seconds:
            _, popped_w = window.pop(0)
            running_sum -= popped_w
        if running_sum > ceiling + 1e-9:
            raise AssertionError(
                f"{label} ceiling violated: {running_sum:.2f} within a {window_seconds}s window ending "
                f"at t={t:.3f}s (ceiling={ceiling})"
            )


def _reset_registry_for_testing(limits: dict[str, ModelLimits] | None = None) -> None:
    """Test-only: clears the cached limiter registry and, if given, injects
    a limits dict instead of reading rate_limits.yaml from disk."""
    global _limiters, _loaded_limits
    _limiters = {}
    if limits is not None:
        _loaded_limits = limits


if __name__ == "__main__":

    async def _self_test() -> None:
        bucket = TokenBucket(capacity=3, refill_period_seconds=1.0)
        t0 = time.monotonic()
        for _ in range(3):
            await bucket.acquire(1)
        elapsed_first_three = time.monotonic() - t0
        assert elapsed_first_three < 0.3, f"bucket starts full — first 3 should be ~instant, took {elapsed_first_three:.3f}s"

        t1 = time.monotonic()
        await bucket.acquire(1)
        elapsed_fourth = time.monotonic() - t1
        assert 0.2 < elapsed_fourth < 0.6, f"4th acquire should wait ~0.33s, took {elapsed_fourth:.3f}s"
        print(f"[rate_limiter] PASS — TokenBucket paces correctly (4th acquire waited {elapsed_fourth:.3f}s, expected ~0.33s)")

        try:
            TokenBucket(capacity=0)
            raise AssertionError("expected ValueError for capacity=0")
        except ValueError:
            print("[rate_limiter] PASS — zero-capacity bucket (unfilled Gemini placeholder) fails loud, not silent")

        _reset_registry_for_testing(
            {
                "llama-3.3-70b-versatile": ModelLimits(rpm=30, tpm=12000),
                "llama-3.1-8b-instant": ModelLimits(rpm=30, tpm=6000),
            }
        )
        assert settings.escalation_model == settings.judge_fallback_model == "llama-3.3-70b-versatile"
        escalation_limiter = get_model_limiter(settings.escalation_model)
        judge_fallback_limiter = get_model_limiter(settings.judge_fallback_model)
        assert escalation_limiter is judge_fallback_limiter
        print("[rate_limiter] PASS — roles sharing a physical model share one token bucket (no accidental 2x budget)")

        _reset_registry_for_testing({"llama-3.1-8b-instant": ModelLimits(rpm=30, tpm=6000)})

        @rate_limited("extraction_model", estimate_tokens=10)
        async def _mock_call(i: int) -> int:
            return i

        results = await asyncio.gather(*(_mock_call(i) for i in range(5)))
        assert results == list(range(5))
        print("[rate_limiter] PASS — rate_limited() wrapper gates calls without altering return values")

        violating_timestamps = [0.0, 0.1, 0.2, 0.3]
        try:
            assert_rate_not_exceeded(violating_timestamps, ceiling=3, window_seconds=1.0, label="test")
            raise AssertionError("expected assert_rate_not_exceeded to catch this violation")
        except AssertionError as e:
            assert "ceiling violated" in str(e)
            print("[rate_limiter] PASS — assert_rate_not_exceeded correctly detects a real violation (not just absence-of-error)")

        print(f"\nsettings source: {'real corpmind.config' if _HAS_REAL_SETTINGS else 'stub (corpmind not importable in this environment)'}")
        print("[rate_limiter] ALL CHECKS PASSED")

    asyncio.run(_self_test())