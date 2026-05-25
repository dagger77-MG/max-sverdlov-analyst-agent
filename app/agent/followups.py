from __future__ import annotations

import re
from typing import Any, Callable

from langchain_core.messages import AIMessage

from app.agent.context import latest_user_message
from app.agent.tool_executor import (
    _append_structured_result,
    _append_trace,
    _format_sample_examples_observation,
)
from app.state import AgentState, AnalysisResult


SampleFormatter = Callable[
    [dict[str, str | None], int, int],
    tuple[str, int, int],
]


def _is_more_examples_query(query: str) -> bool:
    """Return True when the user asks for additional examples from prior context."""
    normalized = query.strip().lower()

    if not normalized:
        return False

    more_markers = (
        "more",
        "another",
        "additional",
        "next",
    )
    example_markers = (
        "example",
        "examples",
        "sample",
        "samples",
        "case",
        "cases",
        "row",
        "rows",
    )

    if any(marker in normalized for marker in more_markers) and any(
        marker in normalized for marker in example_markers
    ):
        return True

    return bool(
        re.search(
            r"\b(show|give|list|display)\s+(?:me\s+)?\d+\s+more\b",
            normalized,
        )
        or re.search(
            r"\b(?:show|give|list|display)?\s*(?:me\s+)?(?:another|next|additional)\s+\d+\b",
            normalized,
        )
    )


def _requested_example_count(query: str, default: int = 3) -> int:
    """Extract requested example count from a follow-up query."""
    normalized = query.strip().lower()

    patterns = [
        r"\banother\s+(\d+)\b",
        r"\bmore\s+(\d+)\b",
        r"\bnext\s+(\d+)\b",
        r"\bshow\s+(?:me\s+)?(\d+)\b",
        r"\bgive\s+(?:me\s+)?(\d+)\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, normalized)
        if match:
            return max(1, min(20, int(match.group(1))))

    return default


def _latest_sample_context(
    results: list[AnalysisResult],
) -> tuple[dict[str, str | None], int] | None:
    """Return previous sample filters and next offset for 'show more' follow-ups."""
    for result in reversed(results):
        if result.get("query_type") != "sample":
            continue

        filters = result.get("filters")
        offset = result.get("value")

        if isinstance(filters, dict) and isinstance(offset, int):
            return filters, offset

    return None


def _handle_more_examples_follow_up(
    state: AgentState,
    sample_formatter: SampleFormatter | None = None,
) -> dict[str, Any] | None:
    """Deterministically answer 'show more examples' without asking the LLM."""
    user_query = latest_user_message(state["messages"])

    if not _is_more_examples_query(user_query):
        return None

    sample_context = _latest_sample_context(state["last_structured_results"])
    if sample_context is None:
        return None

    filters, offset = sample_context
    n = _requested_example_count(user_query)

    formatter = sample_formatter or _format_sample_examples_observation
    observation, next_offset, match_count = formatter(
        filters,
        n,
        offset,
    )

    tool_input = {
        **filters,
        "n": n,
        "offset": offset,
    }
    _append_trace(state, "sample_examples", tool_input, observation)
    _append_structured_result(
        state,
        AnalysisResult(
            label="sample_examples",
            value=next_offset,
            query_type="sample",
            filters=filters,
            match_count=match_count,
        ),
    )

    state["final_answer"] = observation

    return {
        "tool_trace": state["tool_trace"],
        "last_structured_results": state["last_structured_results"],
        "final_answer": observation,
        "messages": [AIMessage(content=observation)],
    }