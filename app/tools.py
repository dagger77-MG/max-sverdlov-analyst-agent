from __future__ import annotations

from collections import Counter
from typing import Literal

from pydantic import BaseModel, Field

from app.data_loader import get_dataset_df, get_dataset_metadata


class GetDatasetSchemaInput(BaseModel):
    include_sample_values: bool = Field(
        default=True,
        description="Whether to include a few sample values for each column.",
    )


class DatasetSchemaOutput(BaseModel):
    columns: list[str]
    row_count: int
    sample_values: dict[str, list[str]] | None = None


class FilterRowsInput(BaseModel):
    category: str | None = Field(
        default=None,
        description="Dataset category to filter by.",
    )
    intent: str | None = Field(
        default=None,
        description="Dataset intent to filter by.",
    )
    text_query: str | None = Field(
        default=None,
        description="Case-insensitive text search over instruction and response text.",
    )
    limit: int | None = Field(
        default=None,
        ge=1,
        le=100,
        description="Maximum number of matching row IDs to return.",
    )


class FilterRowsOutput(BaseModel):
    row_ids: list[int]
    match_count: int
    applied_filters: dict[str, str | int | None]


class CountRowsInput(BaseModel):
    row_ids: list[int] | None = Field(
        default=None,
        description="Optional row IDs to count. If omitted, count all rows.",
    )


class CountRowsOutput(BaseModel):
    count: int


class ExampleRow(BaseModel):
    row_id: int
    instruction: str
    response: str | None = None
    category: str | None = None
    intent: str | None = None


class SampleExamplesInput(BaseModel):
    row_ids: list[int] | None = Field(
        default=None,
        description="Optional row IDs to sample from. If omitted, sample from all rows.",
    )
    n: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Number of examples to return.",
    )
    offset: int = Field(
        default=0,
        ge=0,
        description="Offset for follow-up requests like 'show 3 more'.",
    )


class SampleExamplesOutput(BaseModel):
    examples: list[ExampleRow]
    next_offset: int


class GroupCountRow(BaseModel):
    label: str
    count: int


class GroupCountsInput(BaseModel):
    group_by: Literal["category", "intent"] = Field(
        description="Column to group by.",
    )
    row_ids: list[int] | None = Field(
        default=None,
        description="Optional row IDs to group. If omitted, group all rows.",
    )
    top_k: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum number of groups to return.",
    )


class GroupCountsOutput(BaseModel):
    group_by: str
    counts: list[GroupCountRow]


class SummarizeRowsInput(BaseModel):
    row_ids: list[int] = Field(
        description="Rows to summarize.",
    )
    focus: str = Field(
        description="What the summary should focus on.",
    )
    max_examples: int = Field(
        default=100,
        ge=5,
        le=300,
        description="Maximum number of rows to use in the summary.",
    )


class SummarizeRowsOutput(BaseModel):
    summary: str
    row_count_used: int
    focus: str


def _normalize_filter_value(value: str | None) -> str | None:
    if value is None:
        return None

    normalized = value.strip()
    if not normalized:
        return None

    return normalized.lower()


def _subset_by_row_ids(row_ids: list[int] | None):
    df = get_dataset_df()

    if row_ids is None:
        return df

    row_id_set = set(row_ids)
    return df[df["row_id"].isin(row_id_set)]


def get_dataset_schema_impl(
    include_sample_values: bool = True,
) -> DatasetSchemaOutput:
    metadata = get_dataset_metadata(include_sample_values=include_sample_values)
    return DatasetSchemaOutput(**metadata)


def filter_rows_impl(
    category: str | None = None,
    intent: str | None = None,
    text_query: str | None = None,
    limit: int | None = None,
) -> FilterRowsOutput:
    df = get_dataset_df()

    category_filter = _normalize_filter_value(category)
    intent_filter = _normalize_filter_value(intent)
    text_filter = _normalize_filter_value(text_query)

    filtered = df

    if category_filter is not None:
        filtered = filtered[
            filtered["category"].fillna("").astype(str).str.lower() == category_filter
        ]

    if intent_filter is not None:
        filtered = filtered[
            filtered["intent"].fillna("").astype(str).str.lower() == intent_filter
        ]

    if text_filter is not None:
        instruction_text = filtered["instruction"].fillna("").astype(str)
        response_text = filtered["response"].fillna("").astype(str)
        text_mask = (
            instruction_text.str.lower().str.contains(text_filter, regex=False)
            | response_text.str.lower().str.contains(text_filter, regex=False)
        )
        filtered = filtered[text_mask]

    match_count = int(len(filtered))

    if limit is not None:
        filtered = filtered.head(limit)

    return FilterRowsOutput(
        row_ids=filtered["row_id"].astype(int).tolist(),
        match_count=match_count,
        applied_filters={
            "category": category,
            "intent": intent,
            "text_query": text_query,
            "limit": limit,
        },
    )


def count_rows_impl(row_ids: list[int] | None = None) -> CountRowsOutput:
    df = _subset_by_row_ids(row_ids)
    return CountRowsOutput(count=int(len(df)))


def sample_examples_impl(
    row_ids: list[int] | None = None,
    n: int = 3,
    offset: int = 0,
) -> SampleExamplesOutput:
    df = _subset_by_row_ids(row_ids)

    selected = df.iloc[offset : offset + n]

    examples = [
        ExampleRow(
            row_id=int(row["row_id"]),
            instruction="" if row["instruction"] is None else str(row["instruction"]),
            response=None if row["response"] is None else str(row["response"]),
            category=None if row["category"] is None else str(row["category"]),
            intent=None if row["intent"] is None else str(row["intent"]),
        )
        for _, row in selected.iterrows()
    ]

    return SampleExamplesOutput(
        examples=examples,
        next_offset=offset + len(examples),
    )


def group_counts_impl(
    group_by: Literal["category", "intent"],
    row_ids: list[int] | None = None,
    top_k: int = 20,
) -> GroupCountsOutput:
    df = _subset_by_row_ids(row_ids)

    values = (
        df[group_by]
        .fillna("UNKNOWN")
        .astype(str)
        .replace("", "UNKNOWN")
        .tolist()
    )

    counter = Counter(values)
    counts = [
        GroupCountRow(label=label, count=int(count))
        for label, count in counter.most_common(top_k)
    ]

    return GroupCountsOutput(group_by=group_by, counts=counts)


def summarize_rows_impl(
    row_ids: list[int],
    focus: str,
    max_examples: int = 100,
) -> SummarizeRowsOutput:
    df = _subset_by_row_ids(row_ids).head(max_examples)

    if df.empty:
        return SummarizeRowsOutput(
            summary="No matching rows were found, so there is nothing to summarize.",
            row_count_used=0,
            focus=focus,
        )

    category_counts = Counter(
        df["category"].fillna("UNKNOWN").astype(str).replace("", "UNKNOWN").tolist()
    )
    intent_counts = Counter(
        df["intent"].fillna("UNKNOWN").astype(str).replace("", "UNKNOWN").tolist()
    )

    example_instructions = (
        df["instruction"]
        .fillna("")
        .astype(str)
        .str.strip()
        .replace("", None)
        .dropna()
        .head(5)
        .tolist()
    )

    top_categories = ", ".join(
        f"{label} ({count})" for label, count in category_counts.most_common(5)
    )
    top_intents = ", ".join(
        f"{label} ({count})" for label, count in intent_counts.most_common(5)
    )

    examples_text = "\n".join(f"- {example}" for example in example_instructions)

    summary = (
        f"Summary focus: {focus}\n"
        f"Rows reviewed: {len(df)}\n"
        f"Top categories: {top_categories or 'N/A'}\n"
        f"Top intents: {top_intents or 'N/A'}\n"
        f"Representative customer instructions:\n{examples_text or '- N/A'}"
    )

    return SummarizeRowsOutput(
        summary=summary,
        row_count_used=int(len(df)),
        focus=focus,
    )


def get_dataset_schema(input_data: GetDatasetSchemaInput) -> DatasetSchemaOutput:
    return get_dataset_schema_impl(
        include_sample_values=input_data.include_sample_values,
    )


def filter_rows(input_data: FilterRowsInput) -> FilterRowsOutput:
    return filter_rows_impl(
        category=input_data.category,
        intent=input_data.intent,
        text_query=input_data.text_query,
        limit=input_data.limit,
    )


def count_rows(input_data: CountRowsInput) -> CountRowsOutput:
    return count_rows_impl(row_ids=input_data.row_ids)


def sample_examples(input_data: SampleExamplesInput) -> SampleExamplesOutput:
    return sample_examples_impl(
        row_ids=input_data.row_ids,
        n=input_data.n,
        offset=input_data.offset,
    )


def group_counts(input_data: GroupCountsInput) -> GroupCountsOutput:
    return group_counts_impl(
        group_by=input_data.group_by,
        row_ids=input_data.row_ids,
        top_k=input_data.top_k,
    )


def summarize_rows(input_data: SummarizeRowsInput) -> SummarizeRowsOutput:
    return summarize_rows_impl(
        row_ids=input_data.row_ids,
        focus=input_data.focus,
        max_examples=input_data.max_examples,
    )