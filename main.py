# main.py
import asyncio
from pathlib import Path
from corpmind.agents.ingestion import ingest_supplier_feed
from corpmind.utils.batch_runner import run_batch
from corpmind.graph.build_graph import build_graph

async def main():
    file_path = Path('data/raw/clothing_sample.csv')
    print(f"Ingesting {file_path}...")
    
    rows = ingest_supplier_feed(file_path, supplier_id="test")
    raw_rows = [r.model_dump() for r in rows][:3]
    
    print(f"Running batch for {len(raw_rows)} rows...")
    
    # Notice: .compile() YAHAN SE HATA DIYA HAI
    graph = build_graph()
    
    initial_state = {
        "raw_rows": raw_rows,
        "items": [],
        "accepted_items": [],
        "flagged_items": [],
        "audit_log": [],
    }
    
    res = await run_batch(initial_state, graph=graph)
    
    accepted = len(res.get("accepted_items", []))
    flagged = len(res.get("flagged_items", []))
    
    print("\n================ RESULT ================")
    print(f"✅ Accepted Items: {accepted}")
    print(f"🚩 Flagged Items:  {flagged}")
    print("========================================\n")

if __name__ == "__main__":
    asyncio.run(main())