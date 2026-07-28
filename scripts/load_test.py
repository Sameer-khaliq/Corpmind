
import asyncio
import time
import json
from pathlib import Path

from corpmind.utils.batch_runner import run_batch
from corpmind.observability.token_tracker import tracker

async def warm_up(items_sample: list, n: int = 3):
    """Burn a few real calls first so bucket refill/connection overhead
    doesn't pollute the real measurement."""
    await run_batch(items_sample[:n])
    tracker.records.clear()  

async def load_test(feed_path: str, n_items: int = 50):
    items = load_real_supplier_rows(feed_path)[:n_items + 3]  # +3 buffer for warm-up

    # warm-up (excluded from measurement)
    await warm_up(items, n=3)
    real_items = items[3:3 + n_items]

    start = time.monotonic()
    result = await run_batch(real_items)
    elapsed = time.monotonic() - start

    per_item_seconds = elapsed / n_items
    projected_500_seconds = per_item_seconds * 500
    projected_500_minutes = projected_500_seconds / 60

    report = {
        "n_items": n_items,
        "elapsed_seconds": elapsed,
        "per_item_seconds": per_item_seconds,
        "projected_500_minutes": projected_500_minutes,
        "token_summary": tracker.summary(),
        "accepted": len(result.get("accepted_items", [])),
        "flagged": len(result.get("flagged_items", [])),
    }

    Path("logs/day17_load_test_report.json").write_text(json.dumps(report, indent=2))
    return report

if __name__ == "__main__":
    report = asyncio.run(load_test("data/sample_feeds/full_test_feed.csv"))
    print(json.dumps(report, indent=2))