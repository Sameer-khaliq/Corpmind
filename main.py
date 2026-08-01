"""
scripts/debug_extraction.py

Diagnoses the extraction root-cause behind Day 18's matching false-merges:
category always coming out "shirts", and color/material/size/description
frequently null. Prints raw source row alongside extraction output so we
can tell whether the info was present in the source and extraction missed
it (a real extraction bug — prompt/taxonomy fix needed), or the info
genuinely wasn't in the source (expected — enrichment's job, not
extraction's).

Run from project root:
    python scripts/debug_extraction.py
"""

from groq import Groq

from corpmind.config import settings
from corpmind.agents.extraction import run_extraction
from corpmind.agents.ingestion import ingest_supplier_feed

FEED_PATH = "data/sample_feeds/day17_amazon_50.csv"  # adjust if needed


def main():
    client = Groq(api_key=settings.GROQ_API_KEY)
    rows = ingest_supplier_feed(FEED_PATH, supplier_id="supplier_a")[:5]

    for row in rows:
        result = run_extraction(client, [row])
        p = result[0]
        print(f"title={p.title!r}")
        print(f"  category={p.category!r} color={p.color!r} material={p.material!r} "
              f"size={p.size!r} description={p.description!r}")
        print(f"  raw source row: {row}")
        print()


if __name__ == "__main__":
    main()