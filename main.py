"""
diagnose_langgraph.py — run this, paste me the FULL output.

Two things it does:
  1. Introspects your installed langgraph 1.2.9's actual signatures for the
     pieces build_graph.py uses (StateGraph.add_conditional_edges,
     CompiledStateGraph.ainvoke, Send) — so we stop guessing against
     possibly-outdated assumptions about the API.
  2. Runs a MINIMAL toy graph (no Send, no our code at all) and a second
     minimal graph WITH Send, to isolate whether "ainvoke returns None" is
     a general 1.2.9 behavior change or specific to the Send-based
     conditional-edge pattern build_graph.py uses.

Run: uv run python diagnose_langgraph.py
(from wherever — no corpmind imports needed, this is fully standalone)
"""

import asyncio
import inspect

import langgraph
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

import importlib.metadata

print("=" * 70)
try:
    print("langgraph version:", importlib.metadata.version("langgraph"))
except Exception as e:
    print("langgraph version: could not determine (", e, ")")
print("=" * 70)

print("\n--- StateGraph.add_conditional_edges signature ---")
print(inspect.signature(StateGraph.add_conditional_edges))

print("\n--- StateGraph.add_node signature ---")
print(inspect.signature(StateGraph.add_node))

print("\n--- StateGraph.compile signature ---")
print(inspect.signature(StateGraph.compile))

print("\n--- RetryPolicy fields (tracing_config.py's GROQ_RETRY_POLICY etc use these) ---")
try:
    from langgraph.types import RetryPolicy

    print("RetryPolicy fields:", getattr(RetryPolicy, "__dataclass_fields__", None) or getattr(RetryPolicy, "model_fields", None) or inspect.signature(RetryPolicy))
except Exception as e:
    print("Could not introspect RetryPolicy:", e)


# ---------------------------------------------------------------------------
# Test 1: minimal graph, NO Send, NO conditional edges at all
# ---------------------------------------------------------------------------

async def test_minimal_no_send():
    from typing import TypedDict

    class MiniState(TypedDict, total=False):
        x: int
        y: int

    async def node_a(state: MiniState) -> dict:
        return {"y": state.get("x", 0) + 1}

    g = StateGraph(MiniState)
    g.add_node("a", node_a)
    g.add_edge(START, "a")
    g.add_edge("a", END)
    compiled = g.compile()
    print("\n--- CompiledStateGraph.ainvoke signature (from a real compiled graph) ---")
    print(inspect.signature(compiled.ainvoke))

    result = await compiled.ainvoke({"x": 5})
    print("\n[TEST 1: no Send] result:", result, "| type:", type(result))
    return result


# ---------------------------------------------------------------------------
# Test 2: minimal graph WITH a Send-based conditional edge (mirrors
# build_graph.py's actual pattern, stripped down)
# ---------------------------------------------------------------------------

async def test_minimal_with_send():
    from typing import Annotated, TypedDict
    import operator

    class MiniState(TypedDict, total=False):
        items: list
        results: Annotated[list, operator.add]

    async def dispatch_node(state: MiniState) -> dict:
        return {}

    async def per_item_node(item) -> dict:
        return {"results": [item * 2]}

    def route(state: MiniState):
        return [Send("per_item", i) for i in state.get("items", [])]

    g = StateGraph(MiniState)
    g.add_node("dispatch", dispatch_node)
    g.add_node("per_item", per_item_node)
    g.add_edge(START, "dispatch")
    g.add_conditional_edges("dispatch", route, ["per_item"])
    g.add_edge("per_item", END)
    compiled = g.compile()

    result = await compiled.ainvoke({"items": [1, 2, 3]})
    print("\n[TEST 2: with Send] result:", result, "| type:", type(result))
    return result


async def main():
    try:
        await test_minimal_no_send()
    except Exception as e:
        print("\n[TEST 1] RAISED:", type(e).__name__, "-", e)

    try:
        await test_minimal_with_send()
    except Exception as e:
        print("\n[TEST 2] RAISED:", type(e).__name__, "-", e)


asyncio.run(main())