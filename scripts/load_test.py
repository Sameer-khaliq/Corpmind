# scripts/load_test_day17.py
import asyncio
import time
import json
import uuid
from pathlib import Path

from corpmind.agents.ingestion import ingest_supplier_feed
from corpmind.utils.batch_runner import run_batch
from corpmind.observability.token_tracker import tracker

FEED_PATH = Path("data/sample_feeds/day17_amazon_50.csv")


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


async def warm_up(all_raw_products: list, n: int = 3):
    warm_state = build_batch_state(all_raw_products[:n], batch_id=f"warmup-{uuid.uuid4().hex[:8]}")
    await run_batch(warm_state)
    tracker.records.clear()  # reset — warm-up calls don't count


async def load_test(feed_path: Path, n_items: int = 50):
    all_raw = ingest_supplier_feed(feed_path, supplier_id="supplier_a")

    if len(all_raw) < n_items + 3:
        raise ValueError(
            f"Need at least {n_items + 3} rows (3 warm-up + {n_items} real), "
            f"got {len(all_raw)}"
        )

    await warm_up(all_raw, n=3)
    real_raw = all_raw[3:3 + n_items]

    real_state = build_batch_state(real_raw, batch_id=f"day17-{uuid.uuid4().hex[:8]}")

    start = time.monotonic()
    result = await run_batch(real_state)
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