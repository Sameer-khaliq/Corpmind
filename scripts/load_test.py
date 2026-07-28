
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