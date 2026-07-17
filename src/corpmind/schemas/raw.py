from pydantic import BaseModel, Field


class RawProduct(BaseModel):
    """One untouched row from a supplier feed, before any normalization.

    Deliberately column-agnostic: raw_fields keeps whatever columns the file
    had, under their original names. Ingestion never guesses at meaning —
    which column is "the title" is the Extraction agent's job (Day 4),
    not this schema's or the ingestion node's.
    """

    supplier_id: str
    source_row_index: int = Field(
        ..., description="0-based row index in the original file, for traceability back to source"
    )
    source_file: str
    raw_fields: dict[str, str | None] = Field(
        ..., description="original_column_name -> raw string value, exactly as read"
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Non-fatal issues tolerated during ingestion — missing "
        "values, an encoding fallback, an entirely empty column",
    )
