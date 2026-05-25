from __future__ import annotations

import json
import re
from typing import Any

from app.agent.context import latest_user_message
from app.memory import read_user_profile_impl
from app.state import AgentState, AnalysisResult, ToolTraceItem
from app.tools import (
    count_rows_impl,
    get_dataset_schema_impl,
    group_counts_impl,
    resolve_filter_value_impl,
    sample_examples_impl,
    summarize_rows_impl,
)


def _append_trace(
    state: AgentState,
    tool_name: str,
    tool_input: dict[str, Any],
    observation: str,
) -> None:
    """Append one tool/observation step to state."""
    state["tool_trace"].append(
        ToolTraceItem(
            event_type="tool",
            tool_name=tool_name,
            tool_input=tool_input,
            observation=observation,
        )
    )


def _append_tool_error(
    state: AgentState,
    tool_name: str,
    tool_input: dict[str, Any],
    error: str,
    required_next_step: str,
) -> None:
    """Record a non-fatal tool contract error as a visible observation."""
    _append_trace(
        state,
        tool_name,
        tool_input,
        _format_model_dict(
            {
                "error": error,
                "required_next_step": required_next_step,
            }
        ),
    )


def _append_structured_result(
    state: AgentState,
    result: AnalysisResult,
    max_results: int = 10,
) -> None:
    """Store compact structured results for follow-up questions."""
    state["last_structured_results"].append(result)
    state["last_structured_results"] = state["last_structured_results"][-max_results:]


def _normalize_tool_input(tool_input: dict[str, Any]) -> dict[str, Any]:
    """Return a shallow JSON-like copy of planner-supplied tool input."""
    return dict(tool_input or {})


def _normalize_resolve_filter_value_input(
    tool_input: dict[str, Any],
) -> dict[str, Any]:
    """Normalize resolver input aliases produced by planner/reviewer models."""
    normalized_input = _normalize_tool_input(tool_input)
    query = (
        normalized_input.get("query")
        or normalized_input.get("filter_value")
        or normalized_input.get("value")
        or ""
    )

    return {
        "query": str(query),
        "columns": normalized_input.get("columns") or ["category", "intent"],
        "top_k": int(normalized_input.get("top_k", 5)),
    }


def _tool_filters(
    normalized_input: dict[str, Any],
) -> dict[str, str | None]:
    """Extract canonical dataset filters from tool input."""
    return {
        "category": normalized_input.get("category"),
        "intent": normalized_input.get("intent"),
        "text_query": normalized_input.get("text_query"),
    }


def _canonical_tool_input(
    tool_name: str,
    tool_input: dict[str, Any],
) -> dict[str, Any]:
    """Return canonical tool input for execution and duplicate checks."""
    normalized_input = _normalize_tool_input(tool_input)

    if tool_name == "resolve_filter_value":
        return _normalize_resolve_filter_value_input(normalized_input)

    if tool_name == "count_rows":
        return _tool_filters(normalized_input)

    if tool_name == "sample_examples":
        return {
            **_tool_filters(normalized_input),
            "n": int(normalized_input.get("n", 3)),
            "offset": int(normalized_input.get("offset", 0)),
        }

    if tool_name == "group_counts":
        return {
            "group_by": normalized_input.get("group_by"),
            **_tool_filters(normalized_input),
            "top_k": int(normalized_input.get("top_k", 20)),
        }

    if tool_name == "summarize_rows":
        return {
            **_tool_filters(normalized_input),
            "focus": normalized_input.get("focus"),
            "target_field": normalized_input.get("target_field", "both"),
            "max_examples": int(normalized_input.get("max_examples", 100)),
        }

    return normalized_input


def _tool_call_already_exists(
    state: AgentState,
    tool_name: str,
    tool_input: dict[str, Any],
) -> bool:
    """Return True when the same tool call already exists in this turn trace."""
    normalized_input = _canonical_tool_input(tool_name, tool_input)
    for item in state["tool_trace"]:
        if item.get("event_type", "tool") != "tool":
            continue

        if item["tool_name"] != tool_name:
            continue

        if _canonical_tool_input(item["tool_name"], item["tool_input"]) == normalized_input:
            return True

    return False


def _format_model_dict(data: dict[str, Any]) -> str:
    """Format tool output as stable JSON for trace observations."""
    return json.dumps(data, ensure_ascii=False, default=str)


def _is_distribution_query(query: str) -> bool:
    """Return True when the user asks for a distribution or breakdown."""
    normalized = query.strip().lower()

    if not normalized:
        return False

    distribution_markers = (
        "distribution",
        "breakdown",
        "group count",
        "group counts",
        "count by",
        "counts by",
        "by category",
        "by intent",
    )

    return any(marker in normalized for marker in distribution_markers)


def _has_explicit_top_k_request(query: str) -> bool:
    """Return True when the user explicitly asks for a limited top-N result."""
    normalized = query.strip().lower()

    if not normalized:
        return False

    return bool(
        re.search(r"\b(top|first|highest|lowest)\s+\d+\b", normalized)
        or "most common" in normalized
        or "least common" in normalized
    )


def _requires_grouped_filtered_scope(query: str, group_by: str) -> bool:
    """Detect grouped questions that require a prior filtered subset."""
    normalized = query.strip().lower()

    if not _is_distribution_query(normalized):
        return False

    if group_by == "intent" and "category" in normalized:
        return True

    if group_by == "category" and "intent" in normalized:
        return True

    return False


def _format_sample_examples_observation(
    filters: dict[str, str | None],
    n: int,
    offset: int,
) -> tuple[str, int, int]:
    """Call sample_examples and format the observation text."""
    result = sample_examples_impl(
        category=filters.get("category"),
        intent=filters.get("intent"),
        text_query=filters.get("text_query"),
        n=n,
        offset=offset,
    )

    example_lines = [
        (
            f"row_id={example.row_id}\n"
            f"category={example.category or ''}\n"
            f"intent={example.intent or ''}\n"
            f"customer_instruction={example.instruction}\n"
            f"support_response={example.response or ''}"
        )
        for example in result.examples
    ]

    observation = (
        f"Returned {len(result.examples)} examples from "
        f"{result.match_count} matching rows. "
        f"Next offset = {result.next_offset}. "
        + ("\n\n" + "\n\n---\n\n".join(example_lines) if example_lines else "")
    )

    return observation, result.next_offset, result.match_count


def _execute_selected_tool(
    state: AgentState,
    tool_name: str,
    tool_input: dict[str, Any],
) -> None:
    """Execute one selected tool and append trace/structured follow-up state."""
    normalized_input = _canonical_tool_input(tool_name, tool_input)

    if tool_name == "get_dataset_schema":
        result = get_dataset_schema_impl(
            include_sample_values=bool(
                normalized_input.get("include_sample_values", True)
            ),
        )
        _append_trace(
            state,
            tool_name,
            normalized_input,
            _format_model_dict(result.model_dump()),
        )
        return

    if tool_name == "resolve_filter_value":
        result = resolve_filter_value_impl(
            query=normalized_input["query"],
            columns=normalized_input["columns"],
            top_k=normalized_input["top_k"],
        )
        _append_trace(
            state,
            tool_name,
            normalized_input,
            _format_model_dict(result.model_dump()),
        )
        _append_structured_result(
            state,
            AnalysisResult(
                label=f"resolve_filter_value:{result.query}",
                value=result.confidence,
                query_type="resolve_filter_value",
                filters=result.recommended_filter,
                match_count=None,
            ),
        )
        return

    if tool_name == "count_rows":
        filters = _tool_filters(normalized_input)
        result = count_rows_impl(**filters)
        _append_trace(
            state,
            tool_name,
            filters,
            _format_model_dict(result.model_dump()),
        )
        _append_structured_result(
            state,
            AnalysisResult(
                label="count_rows",
                value=result.count,
                query_type="count",
                filters=result.applied_filters,
                match_count=result.count,
            ),
        )
        return

    if tool_name == "sample_examples":
        filters = _tool_filters(normalized_input)
        n = int(normalized_input.get("n", 3))
        offset = int(normalized_input.get("offset", 0))
        observation, next_offset, match_count = _format_sample_examples_observation(
            filters=filters,
            n=n,
            offset=offset,
        )
        _append_trace(
            state,
            tool_name,
            {**filters, "n": n, "offset": offset},
            observation,
        )
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
        return

    if tool_name == "group_counts":
        group_by = normalized_input.get("group_by")
        if group_by not in {"category", "intent"}:
            raise ValueError(
                "group_counts requires group_by='category' or group_by='intent'."
            )

        user_query = latest_user_message(state["messages"])
        if (
            _requires_grouped_filtered_scope(user_query, group_by)
            and group_by == "intent"
            and not normalized_input.get("category")
        ):
            _append_tool_error(
                state=state,
                tool_name=tool_name,
                tool_input=normalized_input,
                error=(
                    "group_counts needs a category filter for an intent "
                    "distribution inside a category."
                ),
                required_next_step=(
                    "Resolve the category value, then call group_counts with "
                    "group_by='intent' and category=<resolved_category>."
                ),
            )
            return

        if (
            _requires_grouped_filtered_scope(user_query, group_by)
            and group_by == "category"
            and not normalized_input.get("intent")
        ):
            _append_tool_error(
                state=state,
                tool_name=tool_name,
                tool_input=normalized_input,
                error=(
                    "group_counts needs an intent filter for a category "
                    "distribution inside an intent."
                ),
                required_next_step=(
                    "Resolve the intent value, then call group_counts with "
                    "group_by='category' and intent=<resolved_intent>."
                ),
            )
            return

        top_k = int(normalized_input.get("top_k", 20))
        if (
            _is_distribution_query(user_query)
            and not _has_explicit_top_k_request(user_query)
            and top_k < 20
        ):
            top_k = 20
            normalized_input["top_k"] = top_k

        filters = _tool_filters(normalized_input)
        result = group_counts_impl(
            group_by=group_by,
            **filters,
            top_k=top_k,
        )
        _append_trace(
            state,
            tool_name,
            {"group_by": group_by, **filters, "top_k": top_k},
            _format_model_dict(result.model_dump()),
        )
        _append_structured_result(
            state,
            AnalysisResult(
                label=f"group_counts:{group_by}",
                value=len(result.counts),
                query_type="group_counts",
                filters=result.applied_filters,
                match_count=result.match_count,
            ),
        )
        return

    if tool_name == "summarize_rows":
        filters = _tool_filters(normalized_input)
        focus = str(normalized_input.get("focus") or latest_user_message(state["messages"]))
        normalized_input["focus"] = focus
        result = summarize_rows_impl(
            **filters,
            focus=focus,
            target_field=normalized_input.get("target_field", "both"),
            max_examples=int(normalized_input.get("max_examples", 100)),
        )
        _append_trace(
            state,
            tool_name,
            normalized_input,
            _format_model_dict(result.model_dump()),
        )
        _append_structured_result(
            state,
            AnalysisResult(
                label="summarize_rows",
                value=result.row_count_used,
                query_type="summary",
                filters=result.applied_filters,
                match_count=result.match_count,
            ),
        )
        return

    if tool_name == "read_user_profile":
        profile_user_id = str(normalized_input.get("user_id") or state["user_id"])
        result = read_user_profile_impl(user_id=profile_user_id)
        _append_trace(
            state,
            tool_name,
            normalized_input,
            _format_model_dict(result.model_dump()),
        )
        return

    raise ValueError(f"Unknown tool selected by planner: {tool_name}")