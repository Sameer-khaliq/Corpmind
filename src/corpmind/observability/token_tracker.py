# src/corpmind/observability/token_tracker.py
from dataclasses import dataclass, field
from collections import defaultdict
import time

@dataclass
class TokenTracker:
    # stage -> list of (prompt_tokens, completion_tokens, latency_s)
    records: dict[str, list[tuple[int, int, float]]] = field(default_factory=lambda: defaultdict(list))

    def record(self, stage: str, prompt_tokens: int, completion_tokens: int, latency_s: float):
        self.records[stage].append((prompt_tokens, completion_tokens, latency_s))

    def summary(self) -> dict:
        out = {}
        for stage, rows in self.records.items():
            n = len(rows)
            total_prompt = sum(r[0] for r in rows)
            total_completion = sum(r[1] for r in rows)
            total_latency = sum(r[2] for r in rows)
            out[stage] = {
                "calls": n,
                "avg_prompt_tokens": total_prompt / n,
                "avg_completion_tokens": total_completion / n,
                "avg_total_tokens": (total_prompt + total_completion) / n,
                "avg_latency_s": total_latency / n,
            }
        return out

tracker = TokenTracker()