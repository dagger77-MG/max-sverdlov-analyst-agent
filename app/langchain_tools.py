from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool, StructuredTool

from app.memory import ReadUserProfileInput, read_user_profile_impl
from app.tools import (
    CountRowsInput,
    FilterRowsInput,
    GetDatasetSchemaInput,
    GroupCountsInput,
    SampleExamplesInput,
    SummarizeRowsInput,
    count_rows_impl,
    filter_rows_impl,
    get_dataset_schema_impl,
    group_counts_impl,
    sample_examples_impl,
    summarize_rows_impl,
)


def _get_dataset_schema_tool_func(
    include_sample_values: bool = True,
) -> dict[str, Any]:
    """Return dataset columns, row count, and optional sample values."""
    result = get_dataset_schema_impl(
        include_sample_values=include_sample_values,
    )
    return result.model_dump()


def _filter_rows_tool_func(
    category: str | None = None,
    intent: str | None = None,
    text_query: str | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Filter dataset rows by category, intent, or text query."""
    result = filter_rows_impl(
        category=category,
        intent=intent,
        text_query=text_query,
        limit=limit,
    )
    return result.model_dump()


def _count_rows_tool_func(
    row_ids: list[int] | None = None,
) -> dict[str, Any]:
    """Count all dataset rows or a provided row ID subset."""
    result = count_rows_impl(row_ids=row_ids)
    return result.model_dump()


def _sample_examples_tool_func(
    row_ids: list[int] | None = None,
    n: int = 3,
    offset: int = 0,
) -> dict[str, Any]:
    """Return example rows from the dataset or a filtered subset."""
    result = sample_examples_impl(
        row_ids=row_ids,
        n=n,
        offset=offset,
    )
    return result.model_dump()


def _group_counts_tool_func(
    group_by: str,
    row_ids: list[int] | None = None,
    top_k: int = 20,
) -> dict[str, Any]:
    """Group rows by category or intent and return counts."""
    if group_by not in {"category", "intent"}:
        raise ValueError("group_by must be either 'category' or 'intent'.")

    result = group_counts_impl(
        group_by=group_by,  # type: ignore[arg-type]
        row_ids=row_ids,
        top_k=top_k,
    )
    return result.model_dump()


def _summarize_rows_tool_func(
    row_ids: list[int],
    focus: str,
    max_examples: int = 100,
) -> dict[str, Any]:
    """Summarize selected dataset rows for qualitative analysis."""
    result = summarize_rows_impl(
        row_ids=row_ids,
        focus=focus,
        max_examples=max_examples,
    )
    return result.model_dump()


def _read_user_profile_tool_func(
    user_id: str,
) -> dict[str, Any]:
    """Read the persistent distilled profile for a user."""
    result = read_user_profile_impl(user_id=user_id)
    return result.model_dump()


get_dataset_schema_tool = StructuredTool.from_function(
    name="get_dataset_schema",
    description=(
        "Inspect the Bitext dataset schema. Returns available columns, total row "
        "count, and optional sample values. Use this when the user asks what data "
        "is available or when the dataset structure is unclear."
    ),
    func=_get_dataset_schema_tool_func,
    args_schema=GetDatasetSchemaInput,
)

filter_rows_tool = StructuredTool.from_function(
    name="filter_rows",
    description=(
        "Filter Bitext dataset rows by category, intent, or text query. Returns "
        "matching row_ids, exact match_count, and applied filters. Use this before "
        "counting, sampling examples, grouping, or summarizing a subset."
    ),
    func=_filter_rows_tool_func,
    args_schema=FilterRowsInput,
)

count_rows_tool = StructuredTool.from_function(
    name="count_rows",
    description=(
        "Count all Bitext dataset rows or a complete provided row_id subset. For "
        "filtered counts, prefer filter_rows.match_count when filter_rows already "
        "returned the relevant subset."
    ),
    func=_count_rows_tool_func,
    args_schema=CountRowsInput,
)

sample_examples_tool = StructuredTool.from_function(
    name="sample_examples",
    description=(
        "Return actual example rows from the full dataset or a filtered row_id "
        "subset. Use this when the user asks for examples, samples, cases, or rows. "
        "Use offset for follow-up requests such as 'show me more'."
    ),
    func=_sample_examples_tool_func,
    args_schema=SampleExamplesInput,
)

group_counts_tool = StructuredTool.from_function(
    name="group_counts",
    description=(
        "Group Bitext dataset rows by category or intent and return counts sorted "
        "by frequency. Use this for distributions, most common categories/intents, "
        "or grouped count questions."
    ),
    func=_group_counts_tool_func,
    args_schema=GroupCountsInput,
)

summarize_rows_tool = StructuredTool.from_function(
    name="summarize_rows",
    description=(
        "Summarize selected Bitext dataset rows for qualitative analysis. Use this "
        "for themes, patterns, tone, pain points, or open-ended dataset summaries. "
        "The input must be a relevant row_id subset."
    ),
    func=_summarize_rows_tool_func,
    args_schema=SummarizeRowsInput,
)

read_user_profile_tool = StructuredTool.from_function(
    name="read_user_profile",
    description=(
        "Read the persistent distilled user profile. Use this when the user asks "
        "what the agent remembers about them or when saved preferences are relevant."
    ),
    func=_read_user_profile_tool_func,
    args_schema=ReadUserProfileInput,
)


DATASET_LANGCHAIN_TOOLS: list[BaseTool] = [
    get_dataset_schema_tool,
    filter_rows_tool,
    count_rows_tool,
    sample_examples_tool,
    group_counts_tool,
    summarize_rows_tool,
]

LANGCHAIN_TOOLS: list[BaseTool] = [
    *DATASET_LANGCHAIN_TOOLS,
    read_user_profile_tool,
]


__all__ = [
    "DATASET_LANGCHAIN_TOOLS",
    "LANGCHAIN_TOOLS",
    "count_rows_tool",
    "filter_rows_tool",
    "get_dataset_schema_tool",
    "group_counts_tool",
    "read_user_profile_tool",
    "sample_examples_tool",
    "summarize_rows_tool",
]