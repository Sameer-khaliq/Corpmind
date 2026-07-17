from pathlib import Path

import pandas as pd
import pytest

from corpmind.agents.ingestion import ingest_supplier_feed

SAMPLE_FEEDS = Path(__file__).resolve().parent.parent.parent / "data" / "sample_feeds"


def test_csv_ingestion_preserves_arbitrary_columns(tmp_path):
    csv_path = tmp_path / "supplier_a.csv"
    pd.DataFrame({
        "product_title": ["Blue Cotton Shirt", "Slim Jeans"],
        "brand_name": ["Acme", None],
    }).to_csv(csv_path, index=False)

    rows = ingest_supplier_feed(csv_path, supplier_id="supplier_a")

    assert len(rows) == 2
    assert rows[0].raw_fields["product_title"] == "Blue Cotton Shirt"
    assert rows[1].raw_fields["brand_name"] is None
    assert rows[0].supplier_id == "supplier_a"
    assert rows[0].source_row_index == 0
    assert rows[1].source_row_index == 1


def test_xlsx_ingestion(tmp_path):
    xlsx_path = tmp_path / "supplier_b.xlsx"
    pd.DataFrame({
        "title": ["Leather Handbag"],
        "product_specifications": ['{"color": "red", "material": "leather"}'],
    }).to_excel(xlsx_path, index=False)

    rows = ingest_supplier_feed(xlsx_path, supplier_id="supplier_b")

    assert len(rows) == 1
    assert rows[0].raw_fields["product_specifications"] == '{"color": "red", "material": "leather"}'
    assert "product_title" not in rows[0].raw_fields


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        ingest_supplier_feed("does_not_exist.csv", supplier_id="x")


def test_unsupported_extension_raises(tmp_path):
    bad_path = tmp_path / "feed.txt"
    bad_path.write_text("not a real feed")
    with pytest.raises(ValueError, match="Unsupported file type"):
        ingest_supplier_feed(bad_path, supplier_id="x")


def test_empty_csv_raises(tmp_path):
    csv_path = tmp_path / "empty.csv"
    pd.DataFrame({"col": []}).to_csv(csv_path, index=False)
    with pytest.raises(ValueError, match="no rows"):
        ingest_supplier_feed(csv_path, supplier_id="x")


# --- Day 3 checkpoint, taken literally: the three deliberately messy
# sample feeds in data/sample_feeds/ parse into valid RawProduct lists
# with warnings populated correctly. ---------------------------------------

def test_missing_values_feed_warns_only_on_actually_missing_cells():
    rows = ingest_supplier_feed(SAMPLE_FEEDS / "missing_values.csv", supplier_id="supplier_a")

    assert len(rows) == 3
    assert rows[0].warnings == []  # row 1 is complete — no warning noise
    assert any("brand_name" in w and "price" in w for w in rows[1].warnings)
    assert any("product_title" in w and "sku" in w for w in rows[2].warnings)


def test_empty_column_feed_warns_every_row_once():
    rows = ingest_supplier_feed(SAMPLE_FEEDS / "empty_column.csv", supplier_id="supplier_b")

    assert len(rows) == 3
    for row in rows:
        assert any("notes" in w and "empty across the entire file" in w for w in row.warnings)
        # a file-wide empty column shouldn't also fire the row-level missing-value warning
        assert not any("missing values in columns" in w for w in row.warnings)


def test_mixed_encoding_feed_parses_and_warns():
    rows = ingest_supplier_feed(SAMPLE_FEEDS / "mixed_encoding.csv", supplier_id="supplier_c")

    assert len(rows) == 2
    assert rows[0].raw_fields["product_title"] == "Café Racer Jacket"
    assert rows[1].raw_fields["product_title"] == "Señorita Sandals"
    assert any("encoding" in w for w in rows[0].warnings)
    assert any("encoding" in w for w in rows[1].warnings)
