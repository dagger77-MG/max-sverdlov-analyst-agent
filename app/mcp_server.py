from __future__ import annotations

from typing import Literal

from fastmcp import FastMCP

from app.tools import (
    count_rows_impl,
    filter_rows_impl,
    get_dataset_schema_impl,
    group_counts_impl,
    sample_examples_impl,
    summarize_rows_impl,
)


mcp = FastMCP("bitext-data-tools")


@mcp.tool
def get_dataset_schema(include_sample_values: bool = True) -> dict:
    """Return Bitext dataset columns, row count, and optional sample values."""
    result = get_dataset_schema_impl(include_sample_values=include_sample_values)
    return result.model_dump()


@mcp.tool
def filter_rows(
    category: str | None = None,
    intent: str | None = None,
    text_query: str | None = None,
    limit: int | None = None,
) -> dict:
    """Filter Bitext dataset rows by category, intent, or text query."""
    result = filter_rows_impl(
        category=category,
        intent=intent,
        text_query=text_query,
        limit=limit,
    )
    return result.model_dump()


@mcp.tool
def count_rows(row_ids: list[int] | None = None) -> dict:
    """Count all Bitext dataset rows or a provided list of row IDs."""
    result = count_rows_impl(row_ids=row_ids)
    return result.model_dump()


@mcp.tool
def sample_examples(
    row_ids: list[int] | None = None,
    n: int = 3,
    offset: int = 0,
) -> dict:
    """Return example Bitext rows from the full dataset or a filtered subset."""
    result = sample_examples_impl(
        row_ids=row_ids,
        n=n,
        offset=offset,
    )
    return result.model_dump()


@mcp.tool
def group_counts(
    group_by: Literal["category", "intent"],
    row_ids: list[int] | None = None,
    top_k: int = 20,
) -> dict:
    """Group Bitext dataset rows by category or intent and return counts."""
    result = group_counts_impl(
        group_by=group_by,
        row_ids=row_ids,
        top_k=top_k,
    )
    return result.model_dump()


@mcp.tool
def summarize_rows(
    row_ids: list[int],
    focus: str,
    max_examples: int = 100,
) -> dict:
    """Summarize selected Bitext dataset rows for qualitative analysis."""
    result = summarize_rows_impl(
        row_ids=row_ids,
        focus=focus,
        max_examples=max_examples,
    )
    return result.model_dump()


if __name__ == "__main__":
    mcp.run()