"""Ingestion node: reads a raw supplier feed (CSV or XLSX) into RawProduct
objects, tolerating messy real-world feeds — missing values, non-UTF-8
encodings, an entirely empty column — without crashing, and surfacing what
was tolerated via each row's `warnings` field.

Deliberately does zero schema mapping. Supplier A and Supplier B share no
column names; that's fine here. Mapping column meaning to the fixed schema
is the Extraction agent's job (Day 4), not this node's.
"""
from pathlib import Path

import pandas as pd
from charset_normalizer import from_bytes

from corpmind.schemas.raw import RawProduct

SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}
FALLBACK_ENCODING = "latin-1"


def _detect_encoding(raw_bytes: bytes) -> str:
    """Best-effort encoding detection for CSV bytes. XLSX is a binary
    zip-based format and never goes through this path.

    Restricted to encodings plausible for a Western-European e-commerce
    catalog (utf-8, cp1252, iso-8859-1). Without this restriction,
    charset-normalizer can confidently guess an exotic single-byte
    codepage (e.g. cp1250) on short/ambiguous text — cp1250 and latin-1
    both claim byte 0xF1 but decode it to different characters ('ń' vs
    'ñ'), and on a short sample there isn't enough signal to disambiguate
    without narrowing the candidate set ourselves.
    """
    match = from_bytes(raw_bytes, cp_isolation=["utf_8", "cp1252", "iso8859_1"]).best()
    return (match.encoding if match else None) or "utf-8"


def _read_csv_with_encoding_fallback(file_path: Path) -> tuple[pd.DataFrame, str | None]:
    """Tries utf-8 first (the common case, no warning needed). On failure,
    detects the real encoding and retries; if even that fails, falls back
    to latin-1, which accepts any byte sequence. Returns (df, warning)."""
    try:
        return pd.read_csv(file_path, dtype=str, keep_default_na=True, encoding="utf-8"), None
    except UnicodeDecodeError:
        pass

    raw_bytes = file_path.read_bytes()
    detected = _detect_encoding(raw_bytes)
    try:
        df = pd.read_csv(file_path, dtype=str, keep_default_na=True, encoding=detected)
        return df, f"file read using detected encoding '{detected}' (not utf-8)"
    except (UnicodeDecodeError, LookupError):
        df = pd.read_csv(file_path, dtype=str, keep_default_na=True, encoding=FALLBACK_ENCODING)
        return df, (
            f"file read using fallback encoding '{FALLBACK_ENCODING}' "
            "(utf-8 and detected encoding both failed)"
        )


def _read_dataframe(file_path: Path) -> tuple[pd.DataFrame, str | None]:
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        return _read_csv_with_encoding_fallback(file_path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(file_path, dtype=str), None
    raise ValueError(
        f"Unsupported file type '{suffix}' for {file_path}. "
        f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
    )


def ingest_supplier_feed(file_path: str | Path, supplier_id: str) -> list[RawProduct]:
    """Column-agnostic ingestion with tolerance for messy real-world feeds.

    Raises FileNotFoundError / ValueError early for structural problems
    (missing file, no rows, unsupported type) per the error taxonomy's
    'not retryable, fail fast' rule for config-shaped problems. Everything
    that's tolerable at the row/column level becomes a warning instead.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Supplier feed not found: {path}")

    df, encoding_warning = _read_dataframe(path)
    if df.empty:
        raise ValueError(f"Supplier feed {path} has no rows")

    empty_columns = [col for col in df.columns if df[col].isna().all()]
    file_level_warnings: list[str] = []
    if encoding_warning:
        file_level_warnings.append(encoding_warning)
    for col in empty_columns:
        file_level_warnings.append(f"column '{col}' is empty across the entire file")

    rows: list[RawProduct] = []
    for idx, row in df.iterrows():
        raw_fields = {          # main logic 
            str(col): (None if pd.isna(val) else str(val).strip())
            for col, val in row.items()
        }

        row_warnings = list(file_level_warnings)
        missing_in_row = [col for col, val in raw_fields.items() if val is None]
        row_specific_gaps = sorted(set(missing_in_row) - set(empty_columns))
        if row_specific_gaps:
            row_warnings.append(f"row missing values in columns: {row_specific_gaps}")

        rows.append(
            RawProduct(
                supplier_id=supplier_id,
                source_row_index=int(idx),
                source_file=path.name,
                raw_fields=raw_fields,
                warnings=row_warnings,
            )
        )
    return rows
