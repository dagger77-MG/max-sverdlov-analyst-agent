from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import AIMessage

from app.agent.context import latest_user_message
from app.agent.tool_executor import (
    _normalize_resolve_filter_value_input,
    _requires_grouped_filtered_scope,
)
from app.state import AgentState, ToolTraceItem


def _is_tool_trace_item(trace_item: ToolTraceItem) -> bool:
    """Return True when a visible trace entry is an actual tool event."""
    return trace_item.get("event_type", "tool") == "tool"


def _tool_trace_items(state: AgentState) -> list[ToolTraceItem]:
    """Return only tool events from the mixed visible reasoning trace."""
    return [
        trace_item
        for trace_item in state["tool_trace"]
        if _is_tool_trace_item(trace_item)
    ]


def _append_reviewer_fast_path_trace(
    state: AgentState,
    reason: str,
) -> None:
    """Record when a deterministic fast path intentionally skips reviewer LLM."""
    state["tool_trace"].append(
        {
            "event_type": "reviewer",
            "reviewer_status": "answered",
            "reviewer_reason": reason,
            "reviewer_final_answer": "",
            "suggested_tool_name": "",
            "suggested_tool_input": {},
        }
    )


def _final_answer_update(
    state: AgentState,
    final_answer: str,
) -> dict[str, Any]:
    """Set final_answer and return the standard graph state update."""
    state["final_answer"] = final_answer
    return {
        "tool_trace": state["tool_trace"],
        "last_structured_results": state["last_structured_results"],
        "final_answer": final_answer,
        "messages": [AIMessage(content=final_answer)],
    }


def _is_example_request(query: str) -> bool:
    """Return True when the user asks to show dataset examples/samples/cases."""
    normalized = query.strip().lower()

    if not normalized:
        return False

    action_markers = (
        "show",
        "give",
        "list",
        "display",
        "provide",
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

    return any(marker in normalized for marker in action_markers) and any(
        marker in normalized for marker in example_markers
    )


def _trace_observation_is_error(observation: str) -> bool:
    """Return True when an observation is a tool-contract error payload."""
    try:
        parsed = json.loads(observation)
    except json.JSONDecodeError:
        return False

    return isinstance(parsed, dict) and "error" in parsed


def _failed_explicit_resolver_final_answer(state: AgentState) -> str | None:
    """Return a deterministic cannot-answer after an explicit resolver miss."""
    tool_trace = _tool_trace_items(state)
    if not tool_trace:
        return None

    latest_step = tool_trace[-1]
    if latest_step["tool_name"] != "resolve_filter_value":
        return None

    try:
        observation = json.loads(latest_step["observation"])
    except json.JSONDecodeError:
        return None

    if not isinstance(observation, dict):
        return None

    if observation.get("confidence") != "none":
        return None

    tool_input = _normalize_resolve_filter_value_input(latest_step["tool_input"])
    columns = tool_input["columns"]
    query = tool_input["query"].strip()
    user_query = latest_user_message(state["messages"]).lower()

    if not query:
        return None

    if columns == ["intent"] and "intent" in user_query:
        return (
            f'No matching intent value exists for "{query}" in the dataset, '
            "so I can't answer that request using that intent."
        )

    if columns == ["category"] and "category" in user_query:
        return (
            f'No matching category value exists for "{query}" in the dataset, '
            "so I can't answer that request using that category."
        )

    return None


def _return_failed_explicit_resolver_answer_if_ready(
    state: AgentState,
) -> dict[str, Any] | None:
    """Return immediately after an explicit category/intent resolver miss."""
    final_answer = _failed_explicit_resolver_final_answer(state)
    if final_answer is None:
        return None

    return _final_answer_update(state, final_answer)


def _sample_examples_observation_has_examples(observation: str) -> bool:
    """Return True when a formatted sample_examples observation contains rows."""
    match = re.search(
        r"Returned\s+(\d+)\s+examples\s+from\s+(\d+)\s+matching\s+rows",
        observation,
    )

    if match is None:
        return False

    returned_examples = int(match.group(1))
    matching_rows = int(match.group(2))

    return returned_examples > 0 and matching_rows > 0


def _return_deterministic_sample_examples_answer_if_ready(
    state: AgentState,
    tool_name: str,
) -> dict[str, Any] | None:
    """Return sample_examples output directly only when actual examples exist."""
    if (
        tool_name == "sample_examples"
        and _is_example_request(latest_user_message(state["messages"]))
        and _tool_trace_items(state)
        and _sample_examples_observation_has_examples(
            _tool_trace_items(state)[-1]["observation"]
        )
    ):
        final_answer = _tool_trace_items(state)[-1]["observation"]
        _append_reviewer_fast_path_trace(
            state=state,
            reason=(
                "Reviewer LLM skipped: sample_examples returned requested row "
                "content, so deterministic fast path produced the final answer."
            ),
        )
        return _final_answer_update(state, final_answer)

    return None


def _has_filtered_group_counts_trace(
    state: AgentState,
    group_by: str,
    required_filter_column: str,
) -> bool:
    """Return True when group_counts ran with the required semantic filter."""
    for item in _tool_trace_items(state):
        if item["tool_name"] != "group_counts":
            continue
        if item["tool_input"].get("group_by") != group_by:
            continue
        if not item["tool_input"].get(required_filter_column):
            continue
        if _trace_observation_is_error(item["observation"]):
            continue
        return True
    return False


def _has_resolver_trace_for_column(state: AgentState, column: str) -> bool:
    """Return True when this turn has a resolver call that searched column."""
    for item in _tool_trace_items(state):
        if item["tool_name"] != "resolve_filter_value":
            continue
        columns = item["tool_input"].get("columns") or []
        if column in columns and not _trace_observation_is_error(item["observation"]):
            return True
    return False


def _parse_trace_observation_json(
    trace_item: ToolTraceItem,
) -> dict[str, Any] | None:
    """Parse a trace observation as JSON when possible."""
    try:
        parsed = json.loads(trace_item["observation"])
    except json.JSONDecodeError:
        return None

    if not isinstance(parsed, dict):
        return None

    return parsed


def _normalize_contract_text(value: Any) -> str:
    """Normalize trace text for lightweight contract comparisons."""
    return str(value or "").strip().lower()


def _is_text_query_only_summary(trace_item: ToolTraceItem) -> bool:
    """Return True for summarize_rows calls that used only text_query filtering."""
    if trace_item["tool_name"] != "summarize_rows":
        return False

    tool_input = trace_item["tool_input"]
    return (
        not tool_input.get("category")
        and not tool_input.get("intent")
        and bool(tool_input.get("text_query"))
    )


def _summarize_rows_uses_resolved_filter(
    trace_item: ToolTraceItem,
    recommended_filter: dict[str, Any],
) -> bool:
    """Return True when summarize_rows uses the resolver's semantic filter."""
    if trace_item["tool_name"] != "summarize_rows":
        return False

    tool_input = trace_item["tool_input"]
    category = recommended_filter.get("category")
    intent = recommended_filter.get("intent")

    if category and tool_input.get("category") != category:
        return False

    if intent and tool_input.get("intent") != intent:
        return False

    return bool(category or intent)


def _semantic_summary_contract_error(state: AgentState) -> str | None:
    """Block final answers from text-query-only summaries after resolution.

    If a broad phrase was first summarized with only text_query, and a resolver
    later found a concrete category/intent, the final answer must be based on a
    later summarize_rows call using that resolved semantic filter.
    """
    trace = _tool_trace_items(state)

    for resolver_index, trace_item in enumerate(trace):
        if trace_item["tool_name"] != "resolve_filter_value":
            continue

        observation = _parse_trace_observation_json(trace_item)
        if observation is None:
            continue

        if observation.get("confidence") not in {"medium", "high"}:
            continue

        recommended_filter = observation.get("recommended_filter")
        if not isinstance(recommended_filter, dict):
            continue

        if not recommended_filter.get("category") and not recommended_filter.get("intent"):
            continue

        resolver_query = _normalize_contract_text(
            observation.get("query") or trace_item["tool_input"].get("query")
        )
        if not resolver_query:
            continue

        has_matching_text_query_summary = any(
            _is_text_query_only_summary(summary_item)
            and _normalize_contract_text(summary_item["tool_input"].get("text_query"))
            == resolver_query
            for summary_item in trace
        )
        if not has_matching_text_query_summary:
            continue

        has_later_semantic_summary = any(
            _summarize_rows_uses_resolved_filter(
                summary_item,
                recommended_filter=recommended_filter,
            )
            for summary_item in trace[resolver_index + 1 :]
        )
        if has_later_semantic_summary:
            continue

        filter_parts = [
            f'{column}="{value}"'
            for column, value in recommended_filter.items()
            if value
        ]
        filter_text = ", ".join(filter_parts)
        return (
            f'resolve_filter_value recommended {filter_text} for '
            f'"{resolver_query}", but summarize_rows has not been called with '
            "that resolved semantic filter yet. The earlier text_query-only "
            "summarize_rows call is not valid final evidence."
        )

    return None


def _answer_contract_error(state: AgentState) -> str | None:
    """Block final answers for scoped distributions until scoped grouping exists."""
    user_query = latest_user_message(state["messages"])

    if _requires_grouped_filtered_scope(user_query, group_by="intent"):
        if _has_filtered_group_counts_trace(
            state=state,
            group_by="intent",
            required_filter_column="category",
        ):
            return None

        if _has_resolver_trace_for_column(state, "category"):
            return (
                "The category has been resolved, but group_counts has not been "
                "called with group_by='intent' and that category filter yet."
            )

        return (
            "The user asked for an intent distribution inside a category. "
            "Valid evidence requires resolving the category and calling "
            "group_counts with group_by='intent' and category=<resolved_category>."
        )

    if _requires_grouped_filtered_scope(user_query, group_by="category"):
        if _has_filtered_group_counts_trace(
            state=state,
            group_by="category",
            required_filter_column="intent",
        ):
            return None

        if _has_resolver_trace_for_column(state, "intent"):
            return (
                "The intent has been resolved, but group_counts has not been "
                "called with group_by='category' and that intent filter yet."
            )

        return (
            "The user asked for a category distribution inside an intent. "
            "Valid evidence requires resolving the intent and calling "
            "group_counts with group_by='category' and intent=<resolved_intent>."
        )

    semantic_summary_error = _semantic_summary_contract_error(state)
    if semantic_summary_error:
        return semantic_summary_error

    return None


def _build_final_answer_block_feedback(contract_error: str) -> str:
    """Convert deterministic contract validation into planner feedback."""
    return (
        f"Final answer blocked by deterministic validation: {contract_error} "
        "Do not answer from the current observations. Call the missing tool "
        "with explicit semantic filters."
    )