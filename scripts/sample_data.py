# scripts/build_day17_sample.py
import pandas as pd
from pathlib import Path

RAW_FILE = Path("data/raw/clothing_sample.csv")
OUT_PATH = Path("data/sample_feeds/day17_amazon_50.csv")

def main():
    df = pd.read_csv(RAW_FILE)
    sample = df.sample(n=50, random_state=42)
    sample["supplier_id"] = "supplier_a"  # single supplier for this run

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(OUT_PATH, index=False)

    print(f"Saved 50 rows to {OUT_PATH}")
    print(f"Columns: {list(sample.columns)}")

if __name__ == "__main__":
    main()