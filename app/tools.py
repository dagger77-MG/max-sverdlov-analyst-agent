from __future__ import annotations

from collections import Counter
from functools import lru_cache
from typing import Literal
import re

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from app.config import settings
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
        description=(
            "Dataset category to filter by. Use this to find the matching row "
            "subset before counting, grouping, sampling examples, or summarizing."
        ),
    )
    intent: str | None = Field(
        default=None,
        description=(
         "Dataset intent to filter by. Use this to find the matching row "
         "subset before counting, grouping, sampling examples, or summarizing."
        ),
    )
    text_query: str | None = Field(
        default=None,
        description=(
            "Case-insensitive text search over instruction and response text. "
            "Use this to find rows related to a topic when category or intent is "
            "not enough."
        ),
    )
    limit: int | None = Field(
        default=None,
        ge=1,
        le=100,
        description=(
            "Optional cap on returned matching row IDs. Do not use this to satisfy "
            "requests like 'show 3 examples'. For example/sample requests, call "
            "filter_rows without limit to get the matching subset, then call "
            "sample_examples with n set to the requested number. Use limit only "
            "when intentionally restricting the subset for performance or preview."
        ),
    )


class FilterRowsOutput(BaseModel):
    row_ids: list[int]
    match_count: int
    applied_filters: dict[str, str | int | None]


class ResolveFilterValueInput(BaseModel):
    query: str = Field(
        description=(
            "Natural-language value to resolve against actual dataset category "
            "and/or intent values, for example 'refund requests' or 'shipping'."
        ),
    )
    columns: list[Literal["category", "intent"]] = Field(
        default_factory=lambda: ["category", "intent"],
        description=(
            "Dataset columns to search. Use ['intent'] when the user explicitly "
            "asks for an intent, ['category'] when they explicitly ask for a "
            "category, and both when the wording is broad or ambiguous."
        ),
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of matching candidates to return.",
    )


class FilterValueCandidate(BaseModel):
    column: Literal["category", "intent"]
    value: str
    count: int
    score: float
    reason: str


class ResolveFilterValueOutput(BaseModel):
    query: str
    candidates: list[FilterValueCandidate]
    recommended_filter: dict[str, str | None]
    confidence: Literal["none", "low", "medium", "high"]


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
        description=(
            "Optional row IDs to sample from. Usually this should be the row_ids "
            "returned by filter_rows. If omitted, examples are sampled from all rows."
        ),
    )
    n: int = Field(
        default=3,
        ge=1,
        le=20,
        description=(
            "Number of examples to return. Use this field, not filter_rows.limit, "
            "to satisfy requests like 'show 3 examples', 'give me 5 samples', or "
            "'list 2 cases'."
        ),
    )
    offset: int = Field(
        default=0,
        ge=0,
        description=(
            "Offset for follow-up requests like 'show 3 more'. Use 0 for the "
            "first example request, then reuse the previous next_offset for "
            "additional examples from the same row_ids subset."
        ),
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


_CATEGORY_ALIASES: dict[str, str] = {
    "refund": "refund",
    "refunds": "refund",
    "refund request": "refund",
    "refund requests": "refund",
    "reimbursement": "refund",
    "reimbursements": "refund",
    "reimbursement case": "refund",
    "reimbursement cases": "refund",
    "money back": "refund",
    "guarantee": "refund",
    "feedback": "feedback",
    "product feedback": "feedback",
    "customer feedback": "feedback",
    "complaint": "complaint",
    "complaints": "complaint",
    "contact": "contact",
    "contact support": "contact",
    "customer service": "contact",
    "customer service contact": "contact",
    "contact customer service": "contact",
}


def _normalize_category_filter(value: str | None) -> str | None:
    """Normalize category filter values, including common natural-language aliases."""
    normalized = _normalize_filter_value(value)

    if normalized is None:
        return None

    return _CATEGORY_ALIASES.get(normalized, normalized)


def _tokenize_filter_text(value: str) -> set[str]:
    """Tokenize text for lightweight matching against dataset labels."""
    tokens = {
        token
        for token in re.split(r"[^a-z0-9]+", value.lower())
        if token
    }

    expanded = set(tokens)
    for token in tokens:
        if len(token) > 3 and token.endswith("s"):
            expanded.add(token[:-1])

    return expanded


def _score_filter_candidate(
    query: str,
    column: Literal["category", "intent"],
    candidate_value: str,
) -> tuple[float, str]:
    """Score how well a user phrase matches one concrete dataset value."""
    normalized_query = _normalize_filter_value(query) or ""
    normalized_candidate = _normalize_filter_value(candidate_value) or ""

    if not normalized_query or not normalized_candidate:
        return 0.0, "Empty query or candidate."

    if column == "category":
        alias_value = _normalize_category_filter(normalized_query)
        if alias_value == normalized_candidate:
            return 1.0, "Category alias resolves exactly to this dataset value."

    if normalized_query == normalized_candidate:
        return 1.0, "Exact normalized value match."

    query_tokens = _tokenize_filter_text(normalized_query)
    candidate_phrase = normalized_candidate.replace("_", " ")
    candidate_tokens = _tokenize_filter_text(candidate_phrase)

    if column == "intent" and len(query_tokens) == 1:
        return 0.0, (
            "Single-token query is too broad to resolve to a longer intent value."
        )

    if candidate_phrase in normalized_query:
        return 0.90, "Dataset value appears inside the user phrase."

    if normalized_query in candidate_phrase:
        return 0.85, "User phrase appears inside the dataset value."

    if not query_tokens or not candidate_tokens:
        return 0.0, "No comparable tokens."

    overlap = query_tokens & candidate_tokens
    if not overlap:
        return 0.0, "No token overlap."

    overlap_ratio = len(overlap) / max(len(query_tokens), len(candidate_tokens))
    score = min(0.80, 0.30 + overlap_ratio)

    return score, f"Token overlap: {', '.join(sorted(overlap))}."


def _confidence_from_score(score: float) -> Literal["none", "low", "medium", "high"]:
    if score >= 0.85:
        return "high"
    if score >= 0.60:
        return "medium"
    if score > 0:
        return "low"
    return "none"


def resolve_filter_value_impl(
    query: str,
    columns: list[Literal["category", "intent"]] | None = None,
    top_k: int = 5,
) -> ResolveFilterValueOutput:
    """Resolve a natural-language filter phrase to actual dataset values."""
    df = get_dataset_df()
    selected_columns = columns or ["category", "intent"]

    candidates: list[FilterValueCandidate] = []

    for column in selected_columns:
        if column not in {"category", "intent"}:
            continue

        counts = Counter(
            df[column]
            .fillna("")
            .astype(str)
            .replace("", "UNKNOWN")
            .tolist()
        )

        for value, count in counts.items():
            score, reason = _score_filter_candidate(
                query=query,
                column=column,
                candidate_value=value,
            )
            if score <= 0:
                continue

            candidates.append(
                FilterValueCandidate(
                    column=column,
                    value=str(value),
                    count=int(count),
                    score=round(score, 4),
                    reason=reason,
                )
            )

    candidates = sorted(
        candidates,
        key=lambda candidate: (candidate.score, candidate.count),
        reverse=True,
    )[:top_k]

    recommended_filter: dict[str, str | None] = {
        "category": None,
        "intent": None,
    }

    confidence: Literal["none", "low", "medium", "high"] = "none"

    if candidates:
        best = candidates[0]
        confidence = _confidence_from_score(best.score)
        if confidence in {"medium", "high"}:
            recommended_filter[best.column] = best.value

    return ResolveFilterValueOutput(
        query=query,
        candidates=candidates,
        recommended_filter=recommended_filter,
        confidence=confidence,
    )


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

    category_filter = _normalize_category_filter(category)
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


@lru_cache(maxsize=1)
def get_summarizer_llm():
    """Return a cached OpenAI-compatible chat model for row summarization."""
    if not settings.nebius_api_key:
        raise RuntimeError(
            "NEBIUS_API_KEY is missing. Falling back to deterministic summary."
        )

    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "LLM summarization requires 'langchain-openai'. "
            "Falling back to deterministic summary."
        ) from exc

    return ChatOpenAI(
        model=settings.agent_model,
        api_key=settings.nebius_api_key,
        base_url=settings.nebius_base_url,
        temperature=0,
    )


def _deterministic_rows_summary(df, focus: str) -> str:
    """Return a deterministic fallback summary for selected rows."""
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

    return (
        f"Summary focus: {focus}\n"
        f"Rows reviewed: {len(df)}\n"
        f"Top categories: {top_categories or 'N/A'}\n"
        f"Top intents: {top_intents or 'N/A'}\n"
        f"Representative customer instructions:\n{examples_text or '- N/A'}"
    )


def _rows_to_summary_context(df) -> str:
    """Format selected rows as compact text for grounded LLM summarization."""
    lines: list[str] = []

    for _, row in df.iterrows():
        row_id = int(row["row_id"])
        category = "" if row["category"] is None else str(row["category"])
        intent = "" if row["intent"] is None else str(row["intent"])
        instruction = "" if row["instruction"] is None else str(row["instruction"])
        response = "" if row["response"] is None else str(row["response"])

        lines.append(
            f"row_id={row_id}\n"
            f"category={category}\n"
            f"intent={intent}\n"
            f"customer_instruction={instruction}\n"
            f"support_response={response}"
        )

    return "\n\n---\n\n".join(lines)


def _llm_rows_summary(df, focus: str) -> str:
    """Summarize selected rows with an LLM, grounded only in provided rows."""
    llm = get_summarizer_llm()
    row_context = _rows_to_summary_context(df)

    response = llm.invoke(
        [
            SystemMessage(
                content=(
                    "You summarize rows from the Bitext Customer Service dataset. "
                    "Use only the provided rows. Do not add general knowledge. "
                    "Be concise, concrete, and dataset-grounded. Mention recurring "
                    "themes, customer needs, tone, or support patterns only when "
                    "they are visible in the provided rows."
                )
            ),
            HumanMessage(
                content=(
                    f"Focus: {focus}\n"
                    f"Rows provided: {len(df)}\n\n"
                    f"{row_context}\n\n"
                    "Write a short qualitative summary grounded in these rows."
                )
            ),
        ]
    )

    return str(response.content).strip()


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

    try:
        summary = _llm_rows_summary(df=df, focus=focus)
    except Exception:
        summary = _deterministic_rows_summary(df=df, focus=focus)

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


def resolve_filter_value(
    input_data: ResolveFilterValueInput,
) -> ResolveFilterValueOutput:
    return resolve_filter_value_impl(
        query=input_data.query,
        columns=input_data.columns,
        top_k=input_data.top_k,
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