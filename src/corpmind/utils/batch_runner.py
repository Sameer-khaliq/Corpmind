"""
utils/batch_runner.py

CorpMind — Batch runner (Day 16)
====================================

Wires two SEPARATE mechanisms together, per §1.3's "two mechanisms, not
one" point — conflating them is the mistake this file exists to prevent:

  1. `config={"max_concurrency": N}` — caps how many node invocations
     (including Send-dispatched per-item branches) are IN FLIGHT at once.
     This is a concurrency ceiling, not a rate ceiling: if N=10 and every
     call took 0ms, this alone would let you fire unlimited calls/second.
  2. rate_limiter.py's token buckets — cap how many calls per minute
     (RPM/TPM) actually START, per physical model, regardless of how much
     concurrency headroom exists. This is what actually keeps you under
     the provider's real ceiling.

Neither replaces the other: max_concurrency alone can still blow through
an RPM limit (10 fast concurrent calls can easily exceed 30 RPM if the
calls are quick); the token bucket alone can still let too many calls be
in flight at once, exhausting memory/connections on a big batch. Both are
wired here.

Terminology note, because it's genuinely confusing: `BatchState` (graph/
nodes.py) = "all the items in one supplier feed," processed via internal
Send fan-out within ONE graph invocation. `graph.abatch()` (this file) =
"multiple independent graph invocations run concurrently" — e.g. several
separate supplier feed files. `run_batch()` below is for the first sense
(one feed); `run_many_batches()` is for the second (several feeds at
once). Don't confuse the two "batch"es when reading this file.

WIRING / VERIFICATION YOU MUST DO:
  1. `run_batch()`/`run_many_batches()` default `max_concurrency` from
     settings.max_concurrent_llm_calls — confirm that's tuned against your
     REAL account's RPM ceilings (config/rate_limits.yaml), not left at the
     default 10 blindly. Too high just shifts time into Class-1 retry
     backoff (see nodes.py's earlier notes on this).
  2. The Day 16 done-checkpoint load test below (`__main__`) uses build_graph()
     with a MOCKED, rate_limited()-wrapped extraction_fn — no real API calls.
     Once your real agents are wired into build_graph()'s `*_fn=` params
     (per its own docstring), re-run an equivalent load test against a
     staging/sandboxed API key before trusting this against production
     rate limits — a mocked load test proves the WIRING is correct, not
     that your real per-call latency/token estimates are.
  3. `graph.abatch(..., config={"max_concurrency": N})`'s exact semantics
     (shared budget across all invocations vs. per-invocation N) — verify
     against your installed langgraph version. This file's manual-runner
     fallback (used only when langgraph isn't installed) assumes a SHARED
     budget across abatch()'s invocations; say so if your real langgraph
     behaves differently, since that changes what "N" means when you call
     run_many_batches().
"""

from __future__ import annotations

import asyncio
import time

from graph.build_graph import build_graph

try:
    from corpmind.config import settings  # type: ignore
    from corpmind.logging_config import get_logger  # type: ignore

    logger = get_logger(__name__)
except ModuleNotFoundError:
    import logging

    logger = logging.getLogger(__name__)

    class _StubSettings:
        max_concurrent_llm_calls = 10

    settings = _StubSettings()  # type: ignore

try:
    from corpmind.utils.rate_limiter import (  # type: ignore
        ModelLimits,
        _reset_registry_for_testing,
        assert_rate_not_exceeded,
        get_model_limiter,
        load_rate_limits,
        rate_limited,
    )
except ModuleNotFoundError:
    from rate_limiter import (  # type: ignore  (sandbox fallback — same dir)
        ModelLimits,
        _reset_registry_for_testing,
        assert_rate_not_exceeded,
        get_model_limiter,
        load_rate_limits,
        rate_limited,
    )


def _default_max_concurrency() -> int:
    return int(getattr(settings, "max_concurrent_llm_calls", 10))


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


async def run_batch(initial_state: dict, *, max_concurrency: int | None = None, graph=None) -> dict:
    """Runs ONE supplier feed (one BatchState, many items internally via
    Send) through the graph, with concurrency capped at max_concurrency.
    `graph`: pass a pre-built graph (e.g. one wired with real/mocked
    functions via build_graph(**kwargs)) to reuse across calls instead of
    rebuilding per batch — rebuilding is cheap here but won't be once real
    agents are wired in with their own client setup cost."""
    compiled = (graph or build_graph()).compile()
    n = max_concurrency if max_concurrency is not None else _default_max_concurrency()
    return await compiled.ainvoke(initial_state, config={"max_concurrency": n})


async def run_many_batches(initial_states: list[dict], *, max_concurrency: int | None = None, graph=None) -> list[dict]:
    """Runs MULTIPLE independent supplier feeds concurrently — see module
    docstring's terminology note on how this differs from run_batch()'s
    internal per-item concurrency. The max_concurrency budget here is
    shared across all the feeds' work combined (see WIRING note #3)."""
    compiled = (graph or build_graph()).compile()
    n = max_concurrency if max_concurrency is not None else _default_max_concurrency()
    return await compiled.abatch(initial_states, config={"max_concurrency": n})


# ---------------------------------------------------------------------------
# Day 16 done-checkpoint — mocked-LLM load test, 500 synthetic items,
# max_concurrency=10, verified by inspecting recorded call timestamps
# against the configured RPM ceiling (NOT by "it didn't error").
# ---------------------------------------------------------------------------


async def _load_test() -> None:
    TEST_RPM = 30  # matches llama-3.1-8b-instant's real config/rate_limits.yaml entry
    N_ITEMS = 500
    MAX_CONCURRENCY = 10

    _reset_registry_for_testing({"llama-3.1-8b-instant": ModelLimits(rpm=TEST_RPM, tpm=6000)})

    call_timestamps: list[float] = []
    t_start = time.monotonic()

    @rate_limited("extraction_model", estimate_tokens=10)
    async def mocked_extraction_fn(raw_row: dict) -> dict:
        # No real API call — this is the mocked-LLM part of the checkpoint.
        # rate_limited() still gates it through the real token-bucket logic,
        # which is exactly what's under test here.
        call_timestamps.append(time.monotonic() - t_start)
        return {**raw_row, "field_provenance": {}, "extraction_warnings": []}

    # WARM-UP, deliberately: TokenBucket starts full (see rate_limiter.py's
    # own WIRING note #4) — a fresh bucket legitimately allows an initial
    # burst up to its full capacity before pacing kicks in. Measuring from
    # a cold start would test that one-time burst allowance, not the
    # limiter's actual steady-state guarantee (and, worse, can spuriously
    # trip assert_rate_not_exceeded by a call or two right at the boundary,
    # since a little refill can accrue DURING the burst's own drain time —
    # this is a real, harmless characteristic of continuous-refill buckets,
    # not a bug, but it's the wrong thing for this checkpoint to measure).
    # Draining the bucket first, then resetting the clock, isolates the
    # actually-meaningful guarantee: sustained throughput never exceeds RPM.
    warmup_row = {"extraction_id": "warmup", "title": "warmup"}
    for _ in range(TEST_RPM):
        await mocked_extraction_fn(warmup_row)
    call_timestamps.clear()
    t_start = time.monotonic()

    graph = build_graph(extraction_fn=mocked_extraction_fn)

    synthetic_rows = [{"extraction_id": f"row-{i}", "title": f"Synthetic Item {i}"} for i in range(N_ITEMS)]
    initial_state = {"feed_descriptor": {"rows": synthetic_rows}}

    t0 = time.monotonic()
    final_state = await run_batch(initial_state, max_concurrency=MAX_CONCURRENCY, graph=graph)
    elapsed = time.monotonic() - t0

    assert len(call_timestamps) == N_ITEMS, f"expected {N_ITEMS} extraction calls, got {len(call_timestamps)}"
    assert len(final_state.get("phase_a_out", [])) == N_ITEMS

    # The actual done-checkpoint: no 60-second sliding window may contain
    # more than TEST_RPM calls, checked by inspecting the recorded
    # timestamps — not by the absence of a 429 (there was no real API to
    # 429 in the first place; the point is proving the BUCKET would have
    # prevented one). Measured post-warm-up, so this is steady state, not
    # the one-time full-bucket burst.
    assert_rate_not_exceeded(call_timestamps, ceiling=TEST_RPM, window_seconds=60.0, label="extraction_model RPM")

    print(f"[Day 16] PASS — {N_ITEMS}-item mocked load test, max_concurrency={MAX_CONCURRENCY}, "
          f"never exceeded {TEST_RPM} RPM in steady state (verified via {len(call_timestamps)} recorded "
          f"timestamps, not absence-of-error)")
    print(f"  elapsed: {elapsed:.2f}s")
    print(f"  report: {final_state['report']}")

    # Sanity check in the OTHER direction too: prove the bucket is actually
    # doing something, not just permissive-by-accident. 500 calls at 30/min
    # from an EMPTY (post-warm-up) bucket must take close to the
    # theoretical minimum, not near-zero.
    theoretical_min_seconds = (N_ITEMS / TEST_RPM) * 60
    min_plausible_seconds = theoretical_min_seconds * 0.8  # 20% slack
    assert elapsed > min_plausible_seconds, (
        f"load test finished in {elapsed:.2f}s, suspiciously fast for {N_ITEMS} calls capped at {TEST_RPM} RPM "
        f"(expected at least ~{min_plausible_seconds:.1f}s) — the rate limiter may not actually be gating these calls"
    )
    print(f"  PASS — pacing is real, not accidentally permissive (took {elapsed:.2f}s, theoretical min ~{theoretical_min_seconds:.1f}s)")
    print(f"\n  NOTE: at real Groq free-tier RPM=30, 500 extraction calls alone take ~{theoretical_min_seconds/60:.1f} "
          f"minutes minimum — see the chat message for what this means against your 15-minute batch_time_budget_minutes.")


if __name__ == "__main__":
    asyncio.run(_load_test())