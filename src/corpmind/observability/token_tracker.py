# src/corpmind/observability/token_tracker.py
from dataclasses import dataclass, field
from collections import defaultdict
import time

@dataclass
class TokenTracker:
    # stage -> list of (prompt_tokens, completion_tokens, latency_s)
    records: dict[str, list[tuple[int, int, float]]] = field(default_factory=lambda: defaultdict(list))

    def record(self, stage: str, prompt_tokens: int, completion_tokens: int, latency_s: float, batch_size: int = 1):
        self.records[stage].append((prompt_tokens, completion_tokens, latency_s, batch_size))

def summary(self) -> dict:
    out = {}
    for stage, rows in self.records.items():
        n_calls = len(rows)
        total_items = sum(r[3] for r in rows)
        total_prompt = sum(r[0] for r in rows)
        total_completion = sum(r[1] for r in rows)
        total_latency = sum(r[2] for r in rows)
        out[stage] = {
            "calls": n_calls,
            "total_items": total_items,
            "avg_prompt_tokens_per_item": total_prompt / total_items,
            "avg_completion_tokens_per_item": total_completion / total_items,
            "avg_total_tokens_per_item": (total_prompt + total_completion) / total_items,
            "avg_latency_per_call_s": total_latency / n_calls,
        }
    return out

tracker = TokenTracker()