from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal

GroupingColumn = Literal["category", "intent"]


@dataclass(frozen=True)
class GroupingRequest:
    """Parsed grouping/distribution intent for one user query."""

    is_grouping_request: bool
    requested_group_by: GroupingColumn | None = None
    is_scoped: bool = False
    required_filter_column: GroupingColumn | None = None
    scope_phrase: str | None = None


_EMPTY_GROUPING_REQUEST = GroupingRequest(is_grouping_request=False)

_GLOBAL_SCOPE_WORDS = {
    "all",
    "all data",
    "all rows",
    "complete dataset",
    "data",
    "dataset",
    "entire data",
    "entire dataset",
    "everything",
    "full data",
    "full dataset",
    "rows",
    "table",
    "the data",
    "the dataset",
    "the rows",
    "the table",
    "whole data",
    "whole dataset",
}

_GROUPING_MARKERS = (
    "break down",
    "breakdown",
    "distribution",
    "count by",
    "counts by",
    "group count",
    "group counts",
    "group by",
    "grouped by",
)

_SCOPE_TRAILING_COLUMN_WORDS = re.compile(
    r"\b(?:category|categories|intent|intents)\b\.?$",
    flags=re.IGNORECASE,
)


def _normalize_text(value: str) -> str:
    """Normalize text for deterministic grouping-intent parsing."""
    normalized = value.strip().lower()
    normalized = re.sub(r"[^a-z0-9_]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _normalize_scope_phrase(value: str | None) -> str | None:
    """Return a cleaned scope phrase or None when no concrete scope exists."""
    if value is None:
        return None

    cleaned = value.strip().strip(" .?!:;,'\"")
    cleaned = re.sub(r"^the\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = _SCOPE_TRAILING_COLUMN_WORDS.sub("", cleaned).strip()
    cleaned = cleaned.strip(" .?!:;,'\"")

    return cleaned or None


def _is_global_scope_phrase(scope_phrase: str | None) -> bool:
    """Return True when a scope phrase means the whole dataset."""
    normalized_scope = _normalize_text(scope_phrase or "")
    if not normalized_scope:
        return True

    return normalized_scope in _GLOBAL_SCOPE_WORDS


def _requested_group_by(normalized_query: str) -> GroupingColumn | None:
    """Detect the requested group-by column from a normalized query."""
    category_patterns = (
        r"\bby\s+categor(?:y|ies)\b",
        r"\bdistribution\s+of\s+categories\b",
        r"\bcategor(?:y|ies)\s+distribution\b",
        r"\b(?:what|which|show)\s+categories\b",
    )
    intent_patterns = (
        r"\bby\s+intents?\b",
        r"\bdistribution\s+of\s+intents\b",
        r"\bintents?\s+distribution\b",
        r"\b(?:what|which|show)\s+intents\b",
    )

    for pattern in category_patterns:
        if re.search(pattern, normalized_query):
            return "category"

    for pattern in intent_patterns:
        if re.search(pattern, normalized_query):
            return "intent"

    if re.search(r"\b(?:category|categories)\b", normalized_query):
        return "category"

    if re.search(r"\b(?:intent|intents)\b", normalized_query):
        return "intent"

    return None


def _is_grouping_query(normalized_query: str, group_by: GroupingColumn | None) -> bool:
    """Return True when the query asks for grouping/distribution evidence."""
    if group_by is None or not normalized_query:
        return False

    if any(marker in normalized_query for marker in _GROUPING_MARKERS):
        return True

    if re.search(rf"\b{group_by}\s+distribution\b", normalized_query):
        return True

    plural = "categories" if group_by == "category" else "intents"
    if re.search(rf"\b{plural}\s+distribution\b", normalized_query):
        return True

    scoped_question_pattern = (
        r"\b(?:what|which)\s+(?:categories|intents)\s+"
        r"(?:appear|occur)\s+(?:under|inside|within)\b"
    )
    if re.search(scoped_question_pattern, normalized_query):
        return True

    scoped_show_pattern = (
        r"\bshow\s+(?:categories|intents)\s+"
        r"(?:under|inside|within)\b"
    )
    if re.search(scoped_show_pattern, normalized_query):
        return True

    return False


def _extract_scope_phrase(query: str, group_by: GroupingColumn) -> str | None:
    """Extract the requested category/intent scope phrase, if present."""
    group_words = (
        r"category|categories" if group_by == "category" else r"intent|intents"
    )

    patterns = [
        rf"\bbreak\s+down\s+(.+?)\s+by\s+(?:{group_words})\b",
        rf"\b(?:{group_words})\s+breakdown\s+for\s+(.+)$",
        (
            rf"\bdistribution\s+of\s+(?:{group_words})\s+"
           r"(?:in|inside|within|under)\s+(.+)$"
        ),
        (
            rf"\b(?:what|which)\s+(?:{group_words})\s+"
            r"(?:appear|occur)\s+(?:under|inside|within)\s+(.+)$"
        ),
        rf"\bshow\s+(?:{group_words})\s+(?:under|inside|within)\s+(.+)$",
    ]

    for pattern in patterns:
        match = re.search(pattern, query, flags=re.IGNORECASE)
        if match:
            return _normalize_scope_phrase(match.group(1))

    return None


def _required_filter_column(group_by: GroupingColumn) -> GroupingColumn:
    """Return the semantic filter column needed for a scoped grouping."""
    return "category" if group_by == "intent" else "intent"


def analyze_grouping_request(query: str) -> GroupingRequest:
    """Parse grouping/distribution semantics from a user query.

    The parser intentionally handles only deterministic grouping scope. It does
    not resolve whether the scope phrase is a real dataset value; resolver/tool
    execution remains responsible for that.
    """
    normalized_query = _normalize_text(query)
    group_by = _requested_group_by(normalized_query)

    if not _is_grouping_query(normalized_query, group_by):
        return _EMPTY_GROUPING_REQUEST

    assert group_by is not None
    scope_phrase = _extract_scope_phrase(query, group_by)

    if scope_phrase is None or _is_global_scope_phrase(scope_phrase):
        return GroupingRequest(
            is_grouping_request=True,
            requested_group_by=group_by,
            is_scoped=False,
            required_filter_column=None,
            scope_phrase=None,
        )

    return GroupingRequest(
        is_grouping_request=True,
        requested_group_by=group_by,
        is_scoped=True,
        required_filter_column=_required_filter_column(group_by),
        scope_phrase=scope_phrase,
    )


def required_scope_filter_column(query: str, group_by: str) -> GroupingColumn | None:
    """Return the required semantic filter column for scoped group_counts."""
    request = analyze_grouping_request(query)
    if (
        request.is_grouping_request
        and request.is_scoped
        and request.requested_group_by == group_by
    ):
        return request.required_filter_column

    return None


def requires_scope_filter(query: str, group_by: str) -> bool:
    """Return True when group_counts needs a semantic filter for this query."""
    return required_scope_filter_column(query, group_by) is not None


def is_global_grouping_request(query: str, group_by: str) -> bool:
    """Return True when query asks for an unscoped full-dataset grouping."""
    request = analyze_grouping_request(query)
    return (
        request.is_grouping_request
        and not request.is_scoped
        and request.requested_group_by == group_by
    )


def group_counts_evidence_matches_request(
    query: str,
    tool_input: dict[str, Any],
) -> bool:
    """Return True when group_counts input matches parsed grouping intent."""
    request = analyze_grouping_request(query)

    if not request.is_grouping_request:
        return False

    if tool_input.get("group_by") != request.requested_group_by:
        return False

    if not request.is_scoped:
        return not (
            tool_input.get("category")
            or tool_input.get("intent")
            or tool_input.get("text_query")
        )

    required_filter_column = request.required_filter_column
    return bool(
        required_filter_column
        and tool_input.get(required_filter_column)
    )


def _observation_has_error(observation: str) -> bool:
    """Return True when a tool observation contains an error payload."""
    try:
        parsed = json.loads(observation)
    except json.JSONDecodeError:
        return False

    return isinstance(parsed, dict) and "error" in parsed


def is_valid_group_counts_evidence(
    query: str,
    tool_input: dict[str, Any],
    observation: str,
) -> bool:
    """Return True when a group_counts trace is valid evidence for query."""
    return (
        not _observation_has_error(observation)
        and group_counts_evidence_matches_request(
            query=query,
            tool_input=tool_input,
       )
    )