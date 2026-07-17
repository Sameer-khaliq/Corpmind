"""Loads the controlled category vocabulary from config/taxonomy.yaml.

Kept separate from config.py so schemas can import just this, without
pulling in the full Settings object (which requires API keys to be set).
"""
from functools import lru_cache
from pathlib import Path

import yaml

DEFAULT_TAXONOMY_PATH = Path("config/taxonomy.yaml")


@lru_cache
def load_taxonomy(path: Path = DEFAULT_TAXONOMY_PATH) -> frozenset[str]:
    if not path.exists():
        raise FileNotFoundError(
            f"Taxonomy file not found at {path}. "
            "This must exist and be non-empty before any extraction runs."
        )
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    categories = data.get("categories") or []
    if not categories:
        raise ValueError(f"{path} has no categories defined under 'categories:'")
    return frozenset(categories)
