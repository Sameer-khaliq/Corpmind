"""
graph/build_graph.py

CorpMind — Graph assembly (Day 14), async (Day 15's latency decision)
=========================================================================

Same two-Send/join-cycle shape as before, now async end-to-end:

    START
      │
      ▼
  ingestion (batch, once)
      │
      ▼  Send fan-out #1 (per item, async — concurrency capped by config={"max_concurrency": N}, see batch_runner.py)
  extract_and_phase_a  ──► accumulates into phase_a_out
      │
      ▼  JOIN (implicit)
  phase_b_matching (batch, sequential — single writer, not a concurrency target)
      │
      ▼  route_after_matching (conditional edge, Send fan-out #2)
      ├── MATCHED_EXISTING ──────► enrich_and_evaluate  ─┐
      └── NEW_PRODUCT/AMBIGUOUS ─► evaluate_only         ├─► accumulates into eval_out
                                                           │
      ◄────────────────────────────────────────────────────┘
      ▼  JOIN (implicit)
  split_results (batch)
      │
      ▼
  report (batch, once)
      │
      ▼
     END

RetryPolicy (Class 1) attached at add_node() time — unchanged from before.
`phase_b_matching` gets none, deliberately.

WIRING / VERIFICATION YOU MUST DO:
  1. Same caveat as before, now sharper under async: extract_and_phase_a
     combines Class-1-retryable extraction with Class-3a-never-retry Phase A
     in one node. Confirm your langgraph version's RetryPolicy can exclude
     VectorStoreFatalError, or split the node — see nodes.py's module
     docstring for the two options.
  2. `add_node(..., retry_policy=...)` — CONFIRMED against your real installed
     langgraph 1.2.9: `StateGraph.add_node`'s actual signature uses
     `retry_policy=`, not `retry=` (an earlier version of this file had the
     wrong kwarg name — fixed).
  3. Once real agents are wired via `*_fn=`, re-run the timing comparison
     below against REAL API calls (not the artificial asyncio.sleep stubs)
     to get an actual measurement against the 60-90s/100-items target —
     everything here proves the WIRING helps, not the real number.
"""

from __future__ import annotations

import asyncio
import time

from corpmind.graph.edges import route_after_matching
from corpmind.graph.nodes import (
    BatchState,
    make_enrich_and_evaluate_node,
    make_evaluate_only_node,
    make_extract_and_phase_a_node,
    make_ingestion_node,
    make_phase_b_node,
    make_report_node,
    make_split_results_node,
)
from corpmind.graph.tracing_config import GEMINI_RETRY_POLICY, GROQ_RETRY_POLICY, configure_tracing, max_concurrent_calls

try:
    from langgraph.graph import END, START, StateGraph  # type: ignore

    _HAS_LANGGRAPH = True
except ModuleNotFoundError:
    _HAS_LANGGRAPH = False
    START, END = "__start__", "__end__"

    class StateGraph:  # minimal local stand-in — sandbox only
        def __init__(self, state_type):
            self.state_type = state_type
            self.nodes: dict[str, tuple] = {}
            self.edges: list[tuple] = []
            self.conditional_edges: list[tuple] = []

        def add_node(self, name, fn, retry_policy=None):
            self.nodes[name] = (fn, retry_policy)
            return self

        def add_edge(self, a, b):
            self.edges.append((a, b))
            return self

        def add_conditional_edges(self, source, router, targets=None):
            self.conditional_edges.append((source, router, targets))
            return self

        def compile(self):
            return _ManualGraphRunner(self)


def build_graph(
    *,
    ingestion_fn=None,
    extraction_fn=None,
    phase_a_fn=None,
    phase_b_fn=None,
    enrichment_fn=None,
    judge_call_fn=None,
    disambiguation_fn=None,
    report_fn=None,
) -> "StateGraph":
    """
    All params optional — omit any of them and that node keeps nodes.py's
    default stub. This exists so callers (Day 16's batch_runner.py, and
    eventually real production wiring) can inject real or rate-limited
    functions without needing to call add_node() twice for the same name
    (which either errors or silently overwrites, depending on langgraph
    version — neither is something to rely on).
    """
    graph = StateGraph(BatchState)

    ingestion_kwargs = {"ingestion_fn": ingestion_fn} if ingestion_fn else {}
    extract_kwargs = {}
    if extraction_fn:
        extract_kwargs["extraction_fn"] = extraction_fn
    if phase_a_fn:
        extract_kwargs["phase_a_fn"] = phase_a_fn
    phase_b_kwargs = {"phase_b_fn": phase_b_fn} if phase_b_fn else {}
    enrich_kwargs = {}
    if enrichment_fn:
        enrich_kwargs["enrichment_fn"] = enrichment_fn
    if judge_call_fn:
        enrich_kwargs["judge_call_fn"] = judge_call_fn
    if disambiguation_fn:
        enrich_kwargs["disambiguation_fn"] = disambiguation_fn
    evaluate_only_kwargs = {"disambiguation_fn": disambiguation_fn} if disambiguation_fn else {}
    report_kwargs = {"report_fn": report_fn} if report_fn else {}

    graph.add_node("ingestion", make_ingestion_node(**ingestion_kwargs))
    graph.add_node("extract_and_phase_a", make_extract_and_phase_a_node(**extract_kwargs), retry_policy=GROQ_RETRY_POLICY)  # see caveat #1
    graph.add_node("phase_b_matching", make_phase_b_node(**phase_b_kwargs))  # NO retry — Class 3a fail-fast deliberate
    graph.add_node("enrich_and_evaluate", make_enrich_and_evaluate_node(**enrich_kwargs), retry_policy=GEMINI_RETRY_POLICY)
    graph.add_node("evaluate_only", make_evaluate_only_node(**evaluate_only_kwargs), retry_policy=GEMINI_RETRY_POLICY)
    graph.add_node("split_results", make_split_results_node())
    graph.add_node("report", make_report_node(**report_kwargs))

    graph.add_edge(START, "ingestion")
    graph.add_conditional_edges(
        "ingestion",
        lambda state: [
            __import__("graph.edges", fromlist=["Send"]).Send("extract_and_phase_a", item)
            for item in state.get("raw_items", [])
        ],
        ["extract_and_phase_a"],
    )
    graph.add_edge("extract_and_phase_a", "phase_b_matching")
    graph.add_conditional_edges("phase_b_matching", route_after_matching, ["enrich_and_evaluate", "evaluate_only"])
    graph.add_edge("enrich_and_evaluate", "split_results")
    graph.add_edge("evaluate_only", "split_results")
    graph.add_edge("split_results", "report")
    graph.add_edge("report", END)

    return graph


# ---------------------------------------------------------------------------
# Manual async runner — ONLY used when langgraph isn't installed (no pypi
# access in this sandbox). Genuinely concurrent (asyncio.gather), so the
# timing comparison below is a real proof of the async win, not theater. On
# your machine, graph.compile() returns a real langgraph CompiledGraph and
# this class is never touched.
# ---------------------------------------------------------------------------


class _ManualGraphRunner:
    """Sandbox-only stand-in. Supports config={"max_concurrency": N} the
    same way real langgraph does: caps how many node invocations run at
    once. abatch() shares ONE semaphore across all its ainvoke() calls
    (including their internal Send fan-outs) — a combined budget, not a
    separate N per invocation — to mirror what real langgraph's abatch is
    documented to do. Verify this assumption against your installed
    langgraph version; I can't check it live from here."""

    def __init__(self, graph: "StateGraph"):
        self.graph = graph

    async def ainvoke(self, initial_state: dict, config: dict | None = None, *, _semaphore: "asyncio.Semaphore | None" = None) -> dict:
        max_concurrency = (config or {}).get("max_concurrency")
        semaphore = _semaphore if _semaphore is not None else (asyncio.Semaphore(max_concurrency) if max_concurrency else None)

        async def run(name: str, node_state: dict) -> dict:
            fn, _retry = self.graph.nodes[name]
            if semaphore is not None:
                async with semaphore:
                    return await fn(node_state)
            return await fn(node_state)

        state = dict(initial_state)
        state.update(await run("ingestion", state))

        phase_a_results = await asyncio.gather(*[run("extract_and_phase_a", item) for item in state.get("raw_items", [])])
        state["phase_a_out"] = [item for r in phase_a_results for item in r.get("phase_a_out", [])]

        state.update(await run("phase_b_matching", state))

        sends = route_after_matching(state)
        eval_results = await asyncio.gather(*[run(send.node, send.arg) for send in sends])
        state["eval_out"] = [item for r in eval_results for item in r.get("eval_out", [])]

        state.update(await run("split_results", state))
        state.update(await run("report", state))
        return state

    async def abatch(self, initial_states: list[dict], config: dict | None = None) -> list[dict]:
        max_concurrency = (config or {}).get("max_concurrency")
        shared_semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else None
        return await asyncio.gather(*[self.ainvoke(s, config=None, _semaphore=shared_semaphore) for s in initial_states])


async def _main() -> None:
    # configure_tracing()  # WIRING: uncomment once LANGCHAIN_API_KEY is set — skipped here to avoid failing loud in sandbox

    compiled = build_graph().compile()

    # === Day 14 checkpoint — single item flows end-to-end ===================
    single_item_input: BatchState = {  # type: ignore[typeddict-item]
        "feed_descriptor": {
            "rows": [
                {"extraction_id": "row-0", "title": "Men's Cotton Crew Neck T-Shirt", "brand": "ExampleBrand", "color": "navy blue", "price": "19.99"}
            ]
        }
    }
    final_state = await compiled.ainvoke(single_item_input)
    print("DEBUG final_state type:", type(final_state), flush=True)
    print("DEBUG final_state value:", final_state, flush=True)
    assert final_state is not None, (
        "compiled.ainvoke() returned None — this means langgraph's real API "
        "differs from what build_graph() assumes somewhere (Send dispatch, "
        "add_conditional_edges, or the state schema). Check `uv pip show "
        "langgraph` and compare against the graph wiring in build_graph()."
    )
    assert len(final_state.get("phase_a_out", [])) == 1
    assert len(final_state.get("matched_items", [])) == 1
    # single item, index 0 -> stub Phase B assigns NEW_PRODUCT (i % 3 != 2, i % 5 != 4)
    assert final_state["matched_items"][0]["match_result"]["decision"] == "NEW_PRODUCT"
    assert len(final_state.get("eval_out", [])) == 1
    assert final_state["eval_out"][0]["evaluation_record"]["overall_verdict"] == "ACCEPT"
    print("[Day 14] PASS — single item flowed end-to-end through the graph.")
    print("  report:", final_state["report"])

    # === Bonus — proves the async conversion actually reduces latency, not
    # just that it compiles. 60 fake items through the stub latencies. =======
    n_items = 60
    many_items_input: BatchState = {  # type: ignore[typeddict-item]
        "feed_descriptor": {
            "rows": [{"extraction_id": f"row-{i}", "title": f"Item {i}"} for i in range(n_items)]
        }
    }

    t0 = time.perf_counter()
    many_final = await compiled.ainvoke(many_items_input)
    elapsed_concurrent = time.perf_counter() - t0

    # Rough serial-equivalent estimate from the same stub latencies, for
    # comparison — NOT a second real run, just arithmetic on the constants
    # nodes.py uses, so this stays honest about what it's proving.
    from graph.nodes import _STUB_LATENCY_SECONDS

    matched_existing = sum(1 for it in many_final["matched_items"] if it["match_result"]["decision"] == "MATCHED_EXISTING")
    evaluate_only = n_items - matched_existing
    serial_estimate = (
        n_items * _STUB_LATENCY_SECONDS  # extraction
        + n_items * _STUB_LATENCY_SECONDS  # phase A
        + matched_existing * (_STUB_LATENCY_SECONDS * 2)  # enrichment, only MATCHED_EXISTING
    )

    print(f"\n[Bonus] {n_items} items, concurrency cap={max_concurrent_calls()}")
    print(f"  actual concurrent run: {elapsed_concurrent:.2f}s")
    print(f"  naive-serial estimate: {serial_estimate:.2f}s")
    assert elapsed_concurrent < serial_estimate, "async run should be meaningfully faster than serial estimate"
    print(f"  PASS — concurrent run is {serial_estimate / max(elapsed_concurrent, 1e-6):.1f}x faster than serial estimate")
    print(f"  routing check: {matched_existing} MATCHED_EXISTING -> enrichment, {evaluate_only} NEW_PRODUCT/AMBIGUOUS -> evaluate_only")

    print("\nlanggraph installed:", _HAS_LANGGRAPH, "— using", "real StateGraph" if _HAS_LANGGRAPH else "manual async runner stand-in")


if __name__ == "__main__":
    asyncio.run(_main())