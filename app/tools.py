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


class DatasetFilterInput(BaseModel):
    category: str | None = Field(
        default=None,
        description=(
            "Optional dataset category filter. Use only actual dataset values "
            "validated by resolve_filter_value or values already observed from "
            "group_counts."
        ),
    )
    intent: str | None = Field(
        default=None,
        description=(
            "Optional dataset intent filter. Use only actual dataset values "
            "validated by resolve_filter_value or values already observed from "
            "group_counts."
        ),
    )
    text_query: str | None = Field(
        default=None,
        description=(
            "Optional case-insensitive text search over customer instruction "
            "and support response text."
        ),
    )


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


class CountRowsInput(DatasetFilterInput):
    pass


class CountRowsOutput(BaseModel):
    count: int
    applied_filters: dict[str, str | None]


class ExampleRow(BaseModel):
    row_id: int
    instruction: str
    response: str | None = None
    category: str | None = None
    intent: str | None = None


class SampleExamplesInput(DatasetFilterInput):
    n: int = Field(
        default=3,
        ge=1,
        le=20,
        description=(
            "Number of examples to return. Use this field to satisfy requests "
            "like 'show 3 examples', 'give me 5 samples', or 'list 2 cases'."
        ),
    )
    offset: int = Field(
        default=0,
        ge=0,
        description=(
            "Offset within the filtered subset for follow-up requests like "
            "'show 3 more'."
        ),
    )


class SampleExamplesOutput(BaseModel):
    examples: list[ExampleRow]
    next_offset: int
    match_count: int
    applied_filters: dict[str, str | None]


class GroupCountRow(BaseModel):
    label: str
    count: int


class GroupCountsInput(DatasetFilterInput):
    group_by: Literal["category", "intent"] = Field(
        description="Column to group by.",
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
    match_count: int
    applied_filters: dict[str, str | None]


class SummarizeRowsInput(DatasetFilterInput):
    focus: str = Field(
        description="What the summary should focus on.",
    )
    target_field: Literal["instruction", "response", "both"] = Field(
        default="both",
        description=(
            "Which row text to summarize: customer instructions, support "
            "responses, or both."
        ),
    )
    max_examples: int = Field(
        default=100,
        ge=5,
        le=300,
        description="Maximum number of matching rows to use in the summary.",
    )


class SummarizeRowsOutput(BaseModel):
    summary: str
    row_count_used: int
    match_count: int
    focus: str
    target_field: Literal["instruction", "response", "both"]
    applied_filters: dict[str, str | None]


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


def _allow_embedded_category_alias(normalized_value: str) -> bool:
    """Return True only for short filter phrases, not full user questions."""
    tokens = _tokenize_filter_text(normalized_value)
    return len(tokens) <= 5


def _resolve_category_alias_in_text(value: str) -> str | None:
    """Resolve category aliases without letting full questions hijack matching.

        Exact aliases are always allowed. Embedded aliases are intentionally limited
        to short filter phrases such as "people wanting their money back". This
        prevents full analytical questions containing words like "customer service"
        from being incorrectly resolved to the CONTACT category.
    """
    normalized = _normalize_filter_value(value)

    if normalized is None:
        return None

    exact_alias = _CATEGORY_ALIASES.get(normalized)
    if exact_alias is not None:
        return exact_alias

    if not _allow_embedded_category_alias(normalized):
        return None

    for alias, category in sorted(
        _CATEGORY_ALIASES.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        if re.search(rf"\b{re.escape(alias)}\b", normalized):
            return category

    return None


def _normalize_category_filter(value: str | None) -> str | None:
    """Normalize category filter values, including safe natural-language aliases."""
    normalized = _normalize_filter_value(value)

    if normalized is None:
        return None

    return _resolve_category_alias_in_text(normalized) or normalized


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
        alias_value = _resolve_category_alias_in_text(normalized_query)
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


def _canonical_dataset_value(
    column: Literal["category", "intent"],
    normalized_value: str | None,
) -> str | None:
    """Return the dataset's canonical casing for a normalized filter value."""
    if normalized_value is None:
        return None

    df = get_dataset_df()
    values = (
        df[column]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )

    for value in values:
        if value.lower() == normalized_value:
            return value

    return normalized_value

def _effective_filters(
    category: str | None = None,
    intent: str | None = None,
    text_query: str | None = None,
) -> dict[str, str | None]:
    """Return normalized semantic filters used for execution and trace output."""
    normalized_category = _normalize_category_filter(category)
    normalized_intent = _normalize_filter_value(intent)
    normalized_text_query = _normalize_filter_value(text_query)

    return {
        "category": _canonical_dataset_value("category", normalized_category),
        "intent": _canonical_dataset_value("intent", normalized_intent),
        "text_query": normalized_text_query,
    }


def _applied_filters(
    category: str | None = None,
    intent: str | None = None,
    text_query: str | None = None,
) -> dict[str, str | None]:
    return _effective_filters(
        category=category,
        intent=intent,
        text_query=text_query,
    )


def _filter_dataset(
    category: str | None = None,
    intent: str | None = None,
    text_query: str | None = None,
):
    """Return dataset rows matching optional semantic filters.

    Tools own row selection internally. The LLM-facing tool contract never passes
    row ID lists between tools.
    """
    df = get_dataset_df()
    effective_filters = _effective_filters(
        category=category,
        intent=intent,
        text_query=text_query,
    )
    category_filter = _normalize_filter_value(effective_filters["category"])
    intent_filter = _normalize_filter_value(effective_filters["intent"])
    text_filter = effective_filters["text_query"]

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

    return filtered


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


def get_dataset_schema_impl(
    include_sample_values: bool = True,
) -> DatasetSchemaOutput:
    metadata = get_dataset_metadata(include_sample_values=include_sample_values)
    return DatasetSchemaOutput(**metadata)


def count_rows_impl(
    category: str | None = None,
    intent: str | None = None,
    text_query: str | None = None,
) -> CountRowsOutput:
    df = _filter_dataset(
        category=category,
        intent=intent,
        text_query=text_query,
    )
    return CountRowsOutput(
        count=int(len(df)),
        applied_filters=_applied_filters(
            category=category,
            intent=intent,
            text_query=text_query,
        ),
    )


def sample_examples_impl(
    category: str | None = None,
    intent: str | None = None,
    text_query: str | None = None,
    n: int = 3,
    offset: int = 0,
) -> SampleExamplesOutput:
    df = _filter_dataset(
        category=category,
        intent=intent,
        text_query=text_query,
    )
    match_count = int(len(df))
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
        match_count=match_count,
        applied_filters=_applied_filters(
            category=category,
            intent=intent,
            text_query=text_query,
        ),
    )


def group_counts_impl(
    group_by: Literal["category", "intent"],
    category: str | None = None,
    intent: str | None = None,
    text_query: str | None = None,
    top_k: int = 20,
) -> GroupCountsOutput:
    df = _filter_dataset(
        category=category,
        intent=intent,
        text_query=text_query,
    )

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

    return GroupCountsOutput(
        group_by=group_by,
        counts=counts,
        match_count=int(len(df)),
        applied_filters=_applied_filters(
            category=category,
            intent=intent,
            text_query=text_query,
        ),
    )


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


def _representative_text_lines(
    df,
    target_field: Literal["instruction", "response", "both"],
) -> list[str]:
    if target_field == "instruction":
        return (
            df["instruction"]
            .fillna("")
            .astype(str)
            .str.strip()
            .replace("", None)
            .dropna()
            .head(5)
            .map(lambda text: f"- Customer: {text}")
            .tolist()
        )

    if target_field == "response":
        return (
            df["response"]
            .fillna("")
            .astype(str)
            .str.strip()
            .replace("", None)
            .dropna()
            .head(5)
            .map(lambda text: f"- Agent: {text}")
            .tolist()
        )

    lines: list[str] = []
    for _, row in df.head(5).iterrows():
        instruction = "" if row["instruction"] is None else str(row["instruction"])
        response = "" if row["response"] is None else str(row["response"])
        lines.append(f"- Customer: {instruction}\n  Agent: {response}")
    return lines


def _deterministic_rows_summary(
    df,
    focus: str,
    target_field: Literal["instruction", "response", "both"],
) -> str:
    """Return a deterministic fallback summary for selected rows."""
    category_counts = Counter(
        df["category"].fillna("UNKNOWN").astype(str).replace("", "UNKNOWN").tolist()
    )
    intent_counts = Counter(
        df["intent"].fillna("UNKNOWN").astype(str).replace("", "UNKNOWN").tolist()
    )

    top_categories = ", ".join(
        f"{label} ({count})" for label, count in category_counts.most_common(5)
    )
    top_intents = ", ".join(
        f"{label} ({count})" for label, count in intent_counts.most_common(5)
    )

    representative_lines = _representative_text_lines(
        df=df,
        target_field=target_field,
    )
    examples_text = "\n".join(representative_lines)
    examples_label = {
        "instruction": "Representative customer instructions",
        "response": "Representative support responses",
        "both": "Representative customer/support exchanges",
    }[target_field]

    return (
        f"Summary focus: {focus}\n"
        f"Rows reviewed: {len(df)}\n"
        f"Top categories: {top_categories or 'N/A'}\n"
        f"Top intents: {top_intents or 'N/A'}\n"
        f"{examples_label}:\n{examples_text or '- N/A'}"
    )


def _rows_to_summary_context(
    df,
    target_field: Literal["instruction", "response", "both"] = "both",
) -> str:
    """Format selected rows as compact text for grounded LLM summarization."""
    lines: list[str] = []

    for _, row in df.iterrows():
        row_id = int(row["row_id"])
        category = "" if row["category"] is None else str(row["category"])
        intent = "" if row["intent"] is None else str(row["intent"])
        instruction = "" if row["instruction"] is None else str(row["instruction"])
        response = "" if row["response"] is None else str(row["response"])

        base_context = (
            f"row_id={row_id}\n"
            f"category={category}\n"
            f"intent={intent}"
        )

        if target_field == "instruction":
            lines.append(
                f"{base_context}\n"
                f"customer_instruction={instruction}"
            )
        elif target_field == "response":
            lines.append(
                f"{base_context}\n"
                f"support_response={response}"
            )
        else:
            lines.append(
                f"{base_context}\n"
                f"customer_instruction={instruction}\n"
                f"support_response={response}"
            )

    return "\n\n---\n\n".join(lines)


def _strip_thinking_markup(text: str) -> str:
    """Remove model thinking markup from user-facing summaries."""
    cleaned = re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return cleaned.strip()


def _llm_rows_summary(
    df,
    focus: str,
    target_field: Literal["instruction", "response", "both"],
) -> str:
    """Summarize selected rows with an LLM, grounded only in provided rows."""
    llm = get_summarizer_llm()
    row_context = _rows_to_summary_context(df, target_field)

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

    return _strip_thinking_markup(str(response.content))


def summarize_rows_impl(
    category: str | None = None,
    intent: str | None = None,
    text_query: str | None = None,
    focus: str = "Summarize the selected rows.",
    max_examples: int = 100,
    target_field: Literal["instruction", "response", "both"] = "both",
) -> SummarizeRowsOutput:
    filtered = _filter_dataset(
        category=category,
        intent=intent,
        text_query=text_query,
    )
    match_count = int(len(filtered))
    df = filtered.head(max_examples)
    applied_filters = _applied_filters(
        category=category,
        intent=intent,
        text_query=text_query,
    )

    if df.empty:
        return SummarizeRowsOutput(
            summary="No matching rows were found, so there is nothing to summarize.",
            row_count_used=0,
            match_count=match_count,
            focus=focus,
            target_field=target_field,
            applied_filters=applied_filters,
        )

    try:
        summary = _llm_rows_summary(
            df=df,
            focus=focus,
            target_field=target_field,
        )
        if not summary:
            summary = (
                    "LLM summarization returned an empty response; "
                    "deterministic fallback used.\n\n"
                    + _deterministic_rows_summary(
                df=df,
                focus=focus,
                target_field=target_field,
            )
            )
    except Exception as exc:
        summary = (
                "LLM summarization failed; deterministic fallback used. "
                f"Error: {type(exc).__name__}: {exc}\n\n"
                + _deterministic_rows_summary(
            df=df,
            focus=focus,
            target_field=target_field,
        )
        )

    return SummarizeRowsOutput(
        summary=summary,
        row_count_used=int(len(df)),
        match_count=match_count,
        focus=focus,
        target_field=target_field,
        applied_filters=applied_filters,
    )


def get_dataset_schema(input_data: GetDatasetSchemaInput) -> DatasetSchemaOutput:
    return get_dataset_schema_impl(
        include_sample_values=input_data.include_sample_values,
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
    return count_rows_impl(
        category=input_data.category,
        intent=input_data.intent,
        text_query=input_data.text_query,
    )


def sample_examples(input_data: SampleExamplesInput) -> SampleExamplesOutput:
    return sample_examples_impl(
        category=input_data.category,
        intent=input_data.intent,
        text_query=input_data.text_query,
        n=input_data.n,
        offset=input_data.offset,
    )


def group_counts(input_data: GroupCountsInput) -> GroupCountsOutput:
    return group_counts_impl(
        group_by=input_data.group_by,
        category=input_data.category,
        intent=input_data.intent,
        text_query=input_data.text_query,
        top_k=input_data.top_k,
    )


def summarize_rows(input_data: SummarizeRowsInput) -> SummarizeRowsOutput:
    return summarize_rows_impl(
        category=input_data.category,
        intent=input_data.intent,
        text_query=input_data.text_query,
        focus=input_data.focus,
        target_field=input_data.target_field,
        max_examples=input_data.max_examples,
    )
