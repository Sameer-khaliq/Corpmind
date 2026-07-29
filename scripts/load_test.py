import asyncio
import time
import json
import uuid
from decimal import Decimal
from pathlib import Path

from groq import Groq

from corpmind.config import settings
from corpmind.agents.ingestion import ingest_supplier_feed
from corpmind.agents.extraction import run_extraction
from corpmind.agents.matching import prepare_batch_index, phase_a_node, phase_b_node
from corpmind.agents.enrichment import enrich_product, fields_needing_enrichment
from corpmind.agents.evaluation import default_judge_call_fn, default_disambiguation_fn
from corpmind.graph.build_graph import build_graph
from corpmind.utils.batch_runner import run_batch
from corpmind.observability.token_tracker import tracker

FEED_PATH = Path("data/sample_feeds/day17_amazon_50.csv")
from corpmind.utils.rate_limiter import rate_limited

class GraphAdapters:
    def __init__(self, client):
        self.client = client
        self._extracted_products = []
        self._batch_index = None

    @rate_limited("extraction_model", estimate_tokens=700.0 * 3)  # worst case: 1 original + 2 reprompts
    async def _call_extraction(self, raw_row):
        return await asyncio.to_thread(run_extraction, self.client, [raw_row])

    async def extraction_fn(self, raw_row):
        result = await self._call_extraction(raw_row)
        product = result[0]
        self._extracted_products.append(product)
        return product

    async def phase_a_fn(self, normalized_product):
        # no model call here — pure retrieval, not rate-limited
        if self._batch_index is None:
            self._batch_index = prepare_batch_index(self._extracted_products)
        item_state = {"item": normalized_product, "batch_index": self._batch_index}
        result = await asyncio.to_thread(phase_a_node, item_state)
        return result["candidate_pairs"]

    async def phase_b_fn(self, with_candidates):
        # no model call here either
        items = [c["normalized_product"] for c in with_candidates]
        all_pairs = []
        for c in with_candidates:
            all_pairs.extend(c["candidates"])
        batch_state = {"items": items, "candidate_pairs": all_pairs}
        result = await asyncio.to_thread(phase_b_node, batch_state)
        match_results = result["match_results"]
        return [
            {**c, "match_result": match_results[str(c["normalized_product"].item_id)]}
            for c in with_candidates
        ]

    @rate_limited(
        "extraction_model",
        estimate_tokens=lambda self, normalized_product: (
            len(fields_needing_enrichment(normalized_product)) * 3 * 600.0
        ),
    )
    async def _call_enrichment(self, normalized_product):
        return await asyncio.to_thread(enrich_product, normalized_product)

    async def enrichment_fn(self, normalized_product, _unused=None):
        result = await self._call_enrichment(normalized_product)
        return {
            "catalog_id": result.catalog_id,
            "field_results": [fr.model_dump() for fr in result.field_results],
        }


def temp_consistency_fn(item: dict):
    """Temporary stub — real consistency-merge logic not yet built.
    No LLM call, doesn't affect Day 17's timing/token numbers."""
    from corpmind.schemas.consistent import ConsistentProduct

    np = item["normalized_product"]
    mr = item["match_result"]

    attributes = {}
    if getattr(np, "color", None):
        attributes["color"] = np.color
    if getattr(np, "material", None):
        attributes["material"] = np.material
    if getattr(np, "size", None):
        attributes["size"] = np.size

    price = np.price if (getattr(np, "price", None) and np.price > 0) else None

    valid_categories = {"casual-shoes", "handbags", "jeans", "shirts", "tops", "tshirts"}
    category = np.category if np.category in valid_categories else "tops"
    # NOTE: items with unresolved/invalid category (extraction failed to
    # determine one) are force-mapped to a placeholder valid category here
    # ONLY so Day 17's timing/token measurement can complete. This is NOT
    # a real data-quality fix — such items should be caught and routed to
    # flagged_items upstream in evaluate_item, not reach consistency_fn
    # with a fabricated category. Flagged as a real gap, not fixed here.

    title = np.title if (np.title and np.title != "UNRESOLVED") else "unknown"

    return ConsistentProduct(
        catalog_id=mr.catalog_id,
        sku=np.sku,
        title=title,
        brand=np.brand,
        category=category,
        description=np.description,
        price=price,
        attributes=attributes,
    )
@rate_limited("judge_model", estimate_tokens=lambda batch: len(batch) * 300.0)
async def rate_limited_judge_call_fn(batch):
    return await asyncio.to_thread(default_judge_call_fn, batch)


@rate_limited("escalation_model", estimate_tokens=400.0)
async def rate_limited_disambiguation_fn(match_result):
    return await asyncio.to_thread(default_disambiguation_fn, match_result)

def build_real_graph(client):
    adapters = GraphAdapters(client)
    graph = build_graph(
        extraction_fn=adapters.extraction_fn,
        phase_a_fn=adapters.phase_a_fn,
        phase_b_fn=adapters.phase_b_fn,
        enrichment_fn=adapters.enrichment_fn,
        judge_call_fn=rate_limited_judge_call_fn,
        disambiguation_fn=rate_limited_disambiguation_fn,
        consistency_fn=temp_consistency_fn,
    )
    return graph
def build_batch_state(raw_products: list, batch_id: str) -> dict:
    items = [
        {
            "raw_row": rp,
            "normalized_product": None,
            "match_result": None,
            "enrichment_result": None,
            "evaluation_record": None,
            "consistent_output": None,
            "audit_entries": [],
            "error": None,
        }
        for rp in raw_products
    ]
    return {
        "batch_id": batch_id,
        "supplier_feeds": [str(FEED_PATH)],
        "items": items,
        "accepted_items": [],
        "flagged_items": [],
        "audit_log": [],
    }


async def warm_up(client, all_raw_products: list, n: int = 3):
    graph = build_real_graph(client)
    warm_state = build_batch_state(all_raw_products[:n], batch_id=f"warmup-{uuid.uuid4().hex[:8]}")
    await run_batch(warm_state, graph=graph)
    tracker.records.clear()  # reset — warm-up calls don't count


async def load_test(feed_path: Path, n_items: int = 50):
    client = Groq(api_key=settings.GROQ_API_KEY)

    all_raw = ingest_supplier_feed(feed_path, supplier_id="supplier_a")
    if len(all_raw) < n_items + 3:
        raise ValueError(
            f"Need at least {n_items + 3} rows (3 warm-up + {n_items} real), got {len(all_raw)}"
        )

    await warm_up(client, all_raw, n=3)
    real_raw = all_raw[3:3 + n_items]

    graph = build_real_graph(client)  # fresh adapters — no state leak from warm-up
    real_state = build_batch_state(real_raw, batch_id=f"day17-{uuid.uuid4().hex[:8]}")

    start = time.monotonic()
    result = await run_batch(real_state, graph=graph)
    elapsed = time.monotonic() - start

    per_item_seconds = elapsed / n_items
    projected_500_minutes = (per_item_seconds * 500) / 60

    report = {
        "n_items": n_items,
        "elapsed_seconds": elapsed,
        "per_item_seconds": per_item_seconds,
        "projected_500_minutes": projected_500_minutes,
        "token_summary": tracker.summary(),
        "accepted": len(result.get("accepted_items", [])),
        "flagged": len(result.get("flagged_items", [])),
    }

    Path("logs").mkdir(exist_ok=True)
    Path("logs/day17_load_test_report.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    report = asyncio.run(load_test(FEED_PATH, n_items=50))
    print(json.dumps(report, indent=2))