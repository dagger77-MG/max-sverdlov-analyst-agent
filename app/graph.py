from __future__ import annotations

import json
import re
import sqlite3
from functools import lru_cache
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from app.config import settings
from app.memory import read_user_profile_impl, update_user_profile_impl
from app.agent.llm_factory import (
    _create_agent_chat_llm,
    get_agent_llm,
    get_structured_observation_reviewer_llm,
    get_structured_profile_llm,
    get_structured_tool_planner_llm,
)
from app.agent.context import (
    _build_planner_messages,
    _build_reviewer_messages,
    _compact_tool_input_for_prompt,
    _compact_tool_trace_for_prompt,
    _latest_user_message,
    _profile_context_for_planner,
    _structured_results_for_prompt,
)
from app.prompts import (
    OUT_OF_SCOPE_REFUSAL,
    PROFILE_UPDATE_SYSTEM_PROMPT,
)
from app.agent.schemas import (
    ObservationReviewDecision,
    PlannerToolName,
    ProfileObservationDecision,
    ToolPlanDecision,
    VALID_PLANNER_TOOL_NAMES,
)

from app.router import RouteDecision, route_query_with_reason
from app.state import AgentState, AnalysisResult, ToolTraceItem
from app.tools import (
    count_rows_impl,
    get_dataset_schema_impl,
    group_counts_impl,
    resolve_filter_value_impl,
    sample_examples_impl,
    summarize_rows_impl,
)


def create_initial_state(
    query: str,
    session_id: str,
    user_id: str,
    max_iterations: int | None = None,
) -> AgentState:
    """Create the initial graph state for one user query."""
    return AgentState(
        messages=[HumanMessage(content=query)],
        session_id=session_id,
        user_id=user_id,
        route=None,
        route_reason=None,
        tool_trace=[],
        last_structured_results=[],
        user_profile="",
        max_iterations=settings.normalize_max_iterations(max_iterations),
        final_answer=None,
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


def _handle_more_examples_follow_up(state: AgentState) -> dict[str, Any] | None:
    """Deterministically answer 'show more examples' without asking the LLM."""
    user_query = _latest_user_message(state["messages"])

    if not _is_more_examples_query(user_query):
        return None

    sample_context = _latest_sample_context(state["last_structured_results"])
    if sample_context is None:
        return None

    filters, offset = sample_context
    n = _requested_example_count(user_query)

    observation, next_offset, match_count = _format_sample_examples_observation(
        filters=filters,
        n=n,
        offset=offset,
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
    return any(
        item["tool_name"] == tool_name
        and _canonical_tool_input(item["tool_name"], item["tool_input"])
        == normalized_input
        for item in state["tool_trace"]
    )


def _format_model_dict(data: dict[str, Any]) -> str:
    """Format tool output as stable JSON for trace observations."""
    return json.dumps(data, ensure_ascii=False, default=str)


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

        user_query = _latest_user_message(state["messages"])
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
        result = summarize_rows_impl(
            **filters,
            focus=str(
                normalized_input.get("focus", _latest_user_message(state["messages"]))
            ),
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


def _trace_observation_is_error(observation: str) -> bool:
    """Return True when an observation is a tool-contract error payload."""
    try:
        parsed = json.loads(observation)
    except json.JSONDecodeError:
        return False

    return isinstance(parsed, dict) and "error" in parsed


def _failed_explicit_resolver_final_answer(state: AgentState) -> str | None:
    """Return a deterministic cannot-answer after an explicit resolver miss."""
    if not state["tool_trace"]:
        return None

    latest_step = state["tool_trace"][-1]
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
    user_query = _latest_user_message(state["messages"]).lower()

    if not query:
        return None

    if columns == ["intent"] and "intent" in user_query:
        return (
            f'No matching intent value exists for "{query}" in the dataset, '
            "so I can't show examples for that intent."
        )

    if columns == ["category"] and "category" in user_query:
        return (
            f'No matching category value exists for "{query}" in the dataset, '
            "so I can't show examples for that category."
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
        and _is_example_request(_latest_user_message(state["messages"]))
        and state["tool_trace"]
        and _sample_examples_observation_has_examples(
            state["tool_trace"][-1]["observation"]
        )
    ):
        final_answer = state["tool_trace"][-1]["observation"]
        return _final_answer_update(state, final_answer)

    return None


def _review_observations(
    state: AgentState,
) -> ObservationReviewDecision:
    """Ask the reviewer whether the current observations answer the query."""
    reviewer_llm = get_structured_observation_reviewer_llm()
    review = reviewer_llm.invoke(_build_reviewer_messages(state))
    if not isinstance(review, ObservationReviewDecision):
        review = ObservationReviewDecision.model_validate(review)
    return review


def _has_filtered_group_counts_trace(
    state: AgentState,
    group_by: str,
    required_filter_column: str,
) -> bool:
    """Return True when group_counts ran with the required semantic filter."""
    for item in state["tool_trace"]:
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
    for item in state["tool_trace"]:
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
    trace = state["tool_trace"]

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
    user_query = _latest_user_message(state["messages"])

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


def _fallback_answer() -> str:
    """Return a safe graph fallback for planner/reviewer/tool errors."""
    return (
        "I could not complete the analysis within the allowed number of "
        "reasoning steps. Please try asking a more specific dataset question."
    )


def _debug_trace_enabled() -> bool:
    """Return True when local live graph-loop debug output is enabled."""
    value = settings.debug_trace
    return value


def _debug_trace(message: str) -> None:
    """Print live graph-loop debug events for local development.

    This is intentionally separate from the user-facing tool_trace. It helps
    debug planner/reviewer loops and swallowed exceptions while the agent is
    still running.
    """
    if _debug_trace_enabled():
        print(f"[debug] {message}", flush=True)


def load_user_profile_node(state: AgentState) -> dict[str, Any]:
    """Load persistent profile memory into graph state."""
    profile = read_user_profile_impl(state["user_id"])
    return {
        "user_profile": profile.profile,
    }


def router_node(state: AgentState) -> dict[str, Any]:
    """Classify the latest user query before tool selection."""
    user_query = _latest_user_message(state["messages"])
    decision: RouteDecision = route_query_with_reason(user_query)

    return {
        "route": decision.route,
        "route_reason": decision.reason,
    }


def refusal_node(state: AgentState) -> dict[str, Any]:
    """Return a scoped refusal for out-of-scope queries."""
    return {
        "final_answer": OUT_OF_SCOPE_REFUSAL,
        "messages": [AIMessage(content=OUT_OF_SCOPE_REFUSAL)],
    }


def data_agent_loop_node(state: AgentState) -> dict[str, Any]:
    """Run a graph-owned plan -> execute -> review loop for dataset questions."""
    deterministic_result = _handle_more_examples_follow_up(state)
    if deterministic_result is not None:
        return deterministic_result

    reviewer_feedback: str | None = None
    reviewer_requires_tool = False

    for iteration_index in range(state["max_iterations"]):
        iteration_number = iteration_index + 1
        try:
            _debug_trace(
                f"iteration {iteration_number}/{state['max_iterations']}: "
                "planner start"
            )
            planner_llm = get_structured_tool_planner_llm()
            plan = planner_llm.invoke(
                _build_planner_messages(state, reviewer_feedback)
            )
            if not isinstance(plan, ToolPlanDecision):
                plan = ToolPlanDecision.model_validate(plan)

            _debug_trace(
                f"iteration {iteration_number}/{state['max_iterations']}: "
                f"planner action={plan.action}; "
                f"tool={plan.tool_name or '-'}; "
                f"reason={plan.reason}"
            )

            if plan.action == "final_answer":
                if reviewer_requires_tool:
                    reviewer_feedback = (
                        "The previous reviewer decision was needs_more, so the "
                        "current observations are not sufficient for a final answer. "
                        "Call exactly one valid next tool. If the user-provided "
                        "category or intent value was not resolved yet, call "
                        "resolve_filter_value with the columns implied by the user's wording."
                    )
                    _debug_trace(
                        f"iteration {iteration_number}/{state['max_iterations']}: "
                        "blocked planner final_answer after reviewer needs_more"
                    )
                    continue

                final_answer = plan.final_answer.strip()
                if not final_answer:
                    reviewer_feedback = (
                        "Planner chose final_answer but returned empty text."
                    )
                    _debug_trace(
                        f"iteration {iteration_number}/{state['max_iterations']}: "
                        "planner returned empty final_answer"
                    )
                    continue
                contract_error = _answer_contract_error(state)
                if contract_error:
                    reviewer_requires_tool = True
                    reviewer_feedback = _build_final_answer_block_feedback(
                        contract_error
                    )
                    _debug_trace(
                        f"iteration {iteration_number}/{state['max_iterations']}: "
                        "blocked planner final_answer by evidence contract"
                    )
                    continue

                state["final_answer"] = final_answer
                _debug_trace(
                    f"iteration {iteration_number}/{state['max_iterations']}: "
                    "returning planner final_answer"
                )
                return _final_answer_update(state, final_answer)

            if (
                plan.action == "call_tool"
                and _tool_call_already_exists(
                    state=state,
                    tool_name=plan.tool_name,
                    tool_input=plan.tool_input,
                )
            ):
                reviewer_requires_tool = False
                reviewer_feedback = (
                    "This exact tool call already exists in the current turn trace. "
                    "Do not repeat it. Use the existing observation to produce a "
                    "final answer or a cannot-answer style final answer."
                )
                _debug_trace(
                    f"iteration {iteration_number}/{state['max_iterations']}: "
                    f"blocked duplicate planner tool call={plan.tool_name}"
                )
                continue

            _execute_selected_tool(
                state=state,
                tool_name=plan.tool_name,
                tool_input=plan.tool_input,
            )
            _debug_trace(
                f"iteration {iteration_number}/{state['max_iterations']}: "
                f"executed tool={plan.tool_name}; "
                f"trace_steps={len(state['tool_trace'])}"
            )

            failed_resolver_answer = (
                _return_failed_explicit_resolver_answer_if_ready(state)
            )
            if failed_resolver_answer is not None:
                _debug_trace(
                    f"iteration {iteration_number}/{state['max_iterations']}: "
                    "returning deterministic failed resolver answer"
                )
                return failed_resolver_answer


            deterministic_sample_answer = (
                _return_deterministic_sample_examples_answer_if_ready(
                    state=state,
                    tool_name=plan.tool_name,
                )
            )
            if deterministic_sample_answer is not None:
                _debug_trace(
                    f"iteration {iteration_number}/{state['max_iterations']}: "
                    "returning deterministic sample_examples answer"
                )
                return deterministic_sample_answer

            _debug_trace(
                f"iteration {iteration_number}/{state['max_iterations']}: "
                "reviewer start"
            )
            try:
                review = _review_observations(state)
            except Exception as exc:
                reviewer_requires_tool = False
                reviewer_feedback = (
                    "The reviewer failed to return a valid structured decision "
                    f"after the latest tool call. Error: {type(exc).__name__}: {exc}. "
                    "Continue agentically from the current tool trace. Do not repeat "
                    "the same failed or already-observed tool call. If the current "
                    "observations fully answer the exact user request, produce a "
                    "grounded final answer. Otherwise, choose exactly one next useful "
                    "tool based on the current observations."
                )
                _debug_trace(
                    f"iteration {iteration_number}/{state['max_iterations']}: "
                    f"reviewer exception={type(exc).__name__}: {exc}; "
                    "continuing with planner feedback"
                )
                continue

            reviewer_feedback = review.reason
            _debug_trace(
                f"iteration {iteration_number}/{state['max_iterations']}: "
                f"reviewer status={review.status}; "
                f"suggested_tool={review.suggested_tool_name or '-'}; "
                f"reason={review.reason}"
            )

            if review.status in {"answered", "cannot_answer"}:
                if review.status == "answered":
                    contract_error = _answer_contract_error(state)
                    if contract_error:
                        reviewer_requires_tool = True
                        reviewer_feedback = _build_final_answer_block_feedback(
                            contract_error
                        )
                        _debug_trace(
                            f"iteration {iteration_number}/{state['max_iterations']}: "
                            "blocked reviewer answered by evidence contract"
                        )
                        continue

                reviewer_requires_tool = False
                final_answer = review.final_answer.strip() or review.reason.strip()
                _debug_trace(
                    f"iteration {iteration_number}/{state['max_iterations']}: "
                    f"returning reviewer status={review.status}"
                )
                return _final_answer_update(state, final_answer)

            if review.status == "needs_more":
                reviewer_requires_tool = True

                if not review.suggested_tool_name:
                    reviewer_feedback = (
                        f"{review.reason}\n"
                        "Reviewer returned needs_more but did not provide a suggested tool. "
                        "Choose exactly one valid next tool yourself. For unresolved "
                        "category/intent filters, use resolve_filter_value first."
                    )
                    _debug_trace(
                        f"iteration {iteration_number}/{state['max_iterations']}: "
                        "reviewer returned needs_more without suggested tool"
                    )
                    continue

                if review.suggested_tool_name not in VALID_PLANNER_TOOL_NAMES:
                    reviewer_requires_tool = False
                    reviewer_feedback = (
                        f"{review.reason}\n"
                        f"Reviewer returned needs_more with invalid suggested_tool_name="
                        f"{review.suggested_tool_name!r}. This is not a callable tool. "
                        "Do not call another tool only because of this malformed reviewer "
                        "decision. If the existing observations are enough, produce a "
                        "final answer. If the requested subset/value does not exist, "
                        "produce a cannot-answer style final answer."
                    )
                    _debug_trace(
                        f"iteration {iteration_number}/{state['max_iterations']}: "
                        f"reviewer suggested invalid tool={review.suggested_tool_name!r}"
                    )
                    continue

                if _tool_call_already_exists(
                    state=state,
                    tool_name=review.suggested_tool_name,
                    tool_input=review.suggested_tool_input,
                ):
                    reviewer_requires_tool = False
                    reviewer_feedback = (
                        f"{review.reason}\n"
                        "The reviewer suggested a tool call that already exists in "
                        "the current turn trace. Do not repeat the same tool call. "
                        "Use the existing observation to produce a final answer or "
                        "a cannot-answer style final answer."
                    )
                    _debug_trace(
                        f"iteration {iteration_number}/{state['max_iterations']}: "
                        "reviewer suggested duplicate tool call"
                    )
                    continue

                _debug_trace(
                    f"iteration {iteration_number}/{state['max_iterations']}: "
                    f"reviewer requested next tool={review.suggested_tool_name}"
                    "\nexecuting directly"
                )

                _execute_selected_tool(
                    state=state,
                    tool_name=review.suggested_tool_name,
                    tool_input=review.suggested_tool_input,
                )

                _debug_trace(
                    f"iteration {iteration_number}/{state['max_iterations']}: "
                    f"executed reviewer tool={review.suggested_tool_name}; "
                    f"trace_steps={len(state['tool_trace'])}"
                )

                failed_resolver_answer = (
                    _return_failed_explicit_resolver_answer_if_ready(state)
                )
                if failed_resolver_answer is not None:
                    _debug_trace(
                        f"iteration {iteration_number}/{state['max_iterations']}: "
                        "returning deterministic reviewer failed resolver answer"
                    )
                    return failed_resolver_answer

                deterministic_sample_answer = (
                    _return_deterministic_sample_examples_answer_if_ready(
                        state=state,
                        tool_name=review.suggested_tool_name,
                    )
                )
                if deterministic_sample_answer is not None:
                    _debug_trace(
                        f"iteration {iteration_number}/{state['max_iterations']}: "
                        "returning deterministic reviewer sample_examples answer"
                    )
                    return deterministic_sample_answer

                reviewer_requires_tool = False
                reviewer_feedback = (
                    f"The reviewer-suggested tool {review.suggested_tool_name} "
                    "has now been executed. The observation reviewer will inspect "
                    "the latest observation before another planner step is allowed."
                )

                _debug_trace(
                    f"iteration {iteration_number}/{state['max_iterations']}: "
                    "reviewer start after direct tool execution"
                )
                try:
                    follow_up_review = _review_observations(state)
                except Exception as exc:
                    reviewer_feedback = (
                        "The reviewer failed to return a valid structured decision "
                        f"after the reviewer-suggested tool. Error: "
                        f"{type(exc).__name__}: {exc}. Continue agentically from "
                        "the current tool trace. Do not repeat the same tool call."
                    )
                    _debug_trace(
                        f"iteration {iteration_number}/{state['max_iterations']}: "
                        f"follow-up reviewer exception={type(exc).__name__}: {exc}; "
                        "continuing with planner feedback"
                    )
                    continue

                reviewer_feedback = follow_up_review.reason
                _debug_trace(
                    f"iteration {iteration_number}/{state['max_iterations']}: "
                    f"follow-up reviewer status={follow_up_review.status}; "
                    f"suggested_tool={follow_up_review.suggested_tool_name or '-'}; "
                    f"reason={follow_up_review.reason}"
                )

                if follow_up_review.status in {"answered", "cannot_answer"}:
                    if follow_up_review.status == "answered":
                        contract_error = _answer_contract_error(state)
                        if contract_error:
                            reviewer_requires_tool = True
                            reviewer_feedback = _build_final_answer_block_feedback(
                                contract_error
                            )
                            _debug_trace(
                                f"iteration {iteration_number}/{state['max_iterations']}: "
                                "blocked follow-up reviewer answered by evidence contract"
                            )
                            continue

                    final_answer = (
                        follow_up_review.final_answer.strip()
                        or follow_up_review.reason.strip()
                    )
                    _debug_trace(
                        f"iteration {iteration_number}/{state['max_iterations']}: "
                        f"returning follow-up reviewer status={follow_up_review.status}"
                    )
                    return _final_answer_update(state, final_answer)

                if follow_up_review.status == "needs_more":
                    reviewer_requires_tool = True
                    reviewer_feedback = (
                        f"{follow_up_review.reason}\n"
                        f"Suggested next tool: {follow_up_review.suggested_tool_name}\n"
                        f"Suggested next input: "
                        f"{json.dumps(_compact_tool_input_for_prompt(follow_up_review.suggested_tool_input), ensure_ascii=False, default=str)}"
                    )
                    continue

        except Exception as exc:
            fallback = _fallback_answer()
            _debug_trace(
                f"iteration {iteration_number}/{state['max_iterations']}: "
                f"exception={type(exc).__name__}: {exc}"
            )
            return _final_answer_update(state, fallback)

    fallback = _fallback_answer()
    _debug_trace(
        f"max_iterations_exhausted={state['max_iterations']}; "
        f"trace_steps={len(state['tool_trace'])}"
    )

    return _final_answer_update(state, fallback)


def profile_update_node(state: AgentState) -> dict[str, Any]:
    """Update the persistent profile only when a durable fact is detected."""
    user_query = _latest_user_message(state["messages"])

    if not user_query.strip():
        return {}

    try:
        profile_llm = get_structured_profile_llm()
        decision = profile_llm.invoke(
            [
                SystemMessage(content=PROFILE_UPDATE_SYSTEM_PROMPT),
                HumanMessage(content=user_query),
            ]
        )
        if not isinstance(decision, ProfileObservationDecision):
            decision = ProfileObservationDecision.model_validate(decision)
    except Exception:
        return {}

    observation = decision.observation.strip()
    if not observation:
        return {}

    updated_profile = update_user_profile_impl(
        user_id=state["user_id"],
        new_observation=observation,
    )
    return {
        "user_profile": updated_profile.profile,
    }


def route_after_router(state: AgentState) -> str:
    """Choose the next graph branch after routing."""
    if state["route"] == "out_of_scope":
        return "refusal_node"

    return "data_agent_loop_node"


@lru_cache(maxsize=1)
def get_checkpointer():
    """Return a persistent SQLite checkpointer for LangGraph state."""
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except ImportError as exc:
        raise RuntimeError(
            "Persistent LangGraph checkpoints require the SQLite checkpoint package. "
            "Install 'langgraph-checkpoint-sqlite' with the project dependencies."
        ) from exc

    settings.ensure_runtime_dirs()
    checkpoint_path = settings.checkpoint_dir / "checkpoint.sqlite"
    connection = sqlite3.connect(str(checkpoint_path), check_same_thread=False)

    checkpointer = SqliteSaver(connection)
    checkpointer.setup()

    return checkpointer


@lru_cache(maxsize=1)
def build_graph():
    """Build and compile the LangGraph agent graph."""
    graph_builder = StateGraph(AgentState)

    graph_builder.add_node("load_user_profile_node", load_user_profile_node)
    graph_builder.add_node("router_node", router_node)
    graph_builder.add_node("data_agent_loop_node", data_agent_loop_node)
    graph_builder.add_node("refusal_node", refusal_node)
    graph_builder.add_node("profile_update_node", profile_update_node)

    graph_builder.add_edge(START, "load_user_profile_node")
    graph_builder.add_edge("load_user_profile_node", "router_node")
    graph_builder.add_conditional_edges(
        "router_node",
        route_after_router,
        {
            "data_agent_loop_node": "data_agent_loop_node",
            "refusal_node": "refusal_node",
        },
    )
    graph_builder.add_edge("data_agent_loop_node", "profile_update_node")
    graph_builder.add_edge("refusal_node", "profile_update_node")
    graph_builder.add_edge("profile_update_node", END)

    return graph_builder.compile(checkpointer=get_checkpointer())


def _build_graph_config(
    session_id: str,
    user_id: str,
    max_iterations: int,
) -> dict[str, Any]:
    """Build LangGraph config, mapping max_iterations to recursion_limit."""
    return {
        "configurable": {
            "thread_id": session_id,
            "user_id": user_id,
        },
        "recursion_limit": max_iterations + 5,
    }


def _create_invocation_state(
    graph,
    query: str,
    session_id: str,
    user_id: str,
    max_iterations: int,
    config: dict[str, Any],
) -> Any:
    """Create graph input while preserving checkpointed follow-up context.

    For a new thread, provide a complete initial state. For an existing
    checkpointed thread, provide only the fields that should reset for the
    current turn plus the new user message. This preserves prior messages and
    recent structured results for follow-up questions.
    """

    try:
        checkpoint_state = graph.get_state(config)
    except Exception:
        return create_initial_state(
            query=query,
            session_id=session_id,
            user_id=user_id,
            max_iterations=max_iterations,
        )

    if not checkpoint_state.values:
        return create_initial_state(
            query=query,
            session_id=session_id,
            user_id=user_id,
            max_iterations=max_iterations,
        )

    return {
        "messages": [HumanMessage(content=query)],
        "session_id": session_id,
        "user_id": user_id,
        "route": None,
        "route_reason": None,
        "tool_trace": [],
        "max_iterations": max_iterations,
        "final_answer": None,
    }


def invoke_agent(
    query: str,
    session_id: str | None = None,
    user_id: str | None = None,
    max_iterations: int | None = None,
) -> AgentState:
    """Invoke the compiled graph for one user query."""
    graph = build_graph()
    normalized_session_id = session_id or settings.default_session_id
    normalized_user_id = user_id or settings.default_user_id
    normalized_max_iterations = settings.normalize_max_iterations(max_iterations)

    config = _build_graph_config(
        session_id=normalized_session_id,
        user_id=normalized_user_id,
        max_iterations=normalized_max_iterations,
    )

    invocation_state = _create_invocation_state(
        graph=graph,
        query=query,
        session_id=normalized_session_id,
        user_id=normalized_user_id,
        max_iterations=normalized_max_iterations,
        config=config,
    )

    return graph.invoke(invocation_state, config=config)