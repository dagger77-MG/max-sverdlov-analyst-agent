from __future__ import annotations

from typing import Literal

from fastmcp import FastMCP

from app.memory import read_user_profile_impl
from app.tools import (
    CountRowsInput,
    GetDatasetSchemaInput,
    GroupCountsInput,
    ResolveFilterValueInput,
    SampleExamplesInput,
    SummarizeRowsInput,
    count_rows_impl,
    get_dataset_schema_impl,
    group_counts_impl,
    resolve_filter_value_impl,
    sample_examples_impl,
    summarize_rows_impl,
)


mcp = FastMCP("bitext-data-tools")


@mcp.tool
def get_dataset_schema(include_sample_values: bool = True) -> dict:
    """Return Bitext dataset columns, row count, and optional sample values."""
    validated = GetDatasetSchemaInput(include_sample_values=include_sample_values)
    result = get_dataset_schema_impl(**validated.model_dump())
    return result.model_dump()


@mcp.tool
def resolve_filter_value(
    query: str,
    columns: list[Literal["category", "intent"]] | None = None,
    top_k: int = 5,
) -> dict:
    """Resolve a phrase to actual Bitext category or intent values."""
    validated = ResolveFilterValueInput(
        query=query,
        columns=columns or ["category", "intent"],
        top_k=top_k,
    )
    result = resolve_filter_value_impl(
        **validated.model_dump(),
    )
    return result.model_dump()


@mcp.tool
def count_rows(
    category: str | None = None,
    intent: str | None = None,
    text_query: str | None = None,
) -> dict:
    """Count all Bitext rows or rows matching semantic filters."""
    validated = CountRowsInput(
        category=category,
        intent=intent,
        text_query=text_query,
    )
    result = count_rows_impl(
        **validated.model_dump(),
    )
    return result.model_dump()


@mcp.tool
def sample_examples(
    category: str | None = None,
    intent: str | None = None,
    text_query: str | None = None,
    n: int = 3,
    offset: int = 0,
) -> dict:
    """Return example Bitext rows matching semantic filters."""
    validated = SampleExamplesInput(
        category=category,
        intent=intent,
        text_query=text_query,
        n=n,
        offset=offset,
    )
    result = sample_examples_impl(
        **validated.model_dump(),
    )
    return result.model_dump()


@mcp.tool
def group_counts(
    group_by: Literal["category", "intent"],
    category: str | None = None,
    intent: str | None = None,
    text_query: str | None = None,
    top_k: int = 20,
) -> dict:
    """Group all Bitext rows or filtered rows by category or intent."""
    validated = GroupCountsInput(
        group_by=group_by,
        category=category,
        intent=intent,
        text_query=text_query,
        top_k=top_k,
    )
    result = group_counts_impl(
        **validated.model_dump(),
    )
    return result.model_dump()


@mcp.tool
def summarize_rows(
    category: str | None = None,
    intent: str | None = None,
    text_query: str | None = None,
    focus: str = "Summarize the selected rows.",
    target_field: Literal["instruction", "response", "both"] = "both",
    max_examples: int = 100,
) -> dict:
    """Summarize Bitext rows matching semantic filters."""
    validated = SummarizeRowsInput(
        category=category,
        intent=intent,
        text_query=text_query,
        focus=focus,
        target_field=target_field,
        max_examples=max_examples,
    )
    result = summarize_rows_impl(
        **validated.model_dump(),
    )
    return result.model_dump()


if __name__ == "__main__":
    mcp.run()