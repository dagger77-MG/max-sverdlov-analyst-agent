from __future__ import annotations

import json
import re
import sqlite3
from functools import lru_cache
from typing import Any, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from app.config import settings
from app.logging_utils import summarize_row_ids
from app.memory import read_user_profile_impl, update_user_profile_impl
from app.prompts import (
    OUT_OF_SCOPE_REFUSAL,
    PROFILE_UPDATE_SYSTEM_PROMPT,
    PLANNER_SYSTEM_PROMPT,
    REVIEWER_SYSTEM_PROMPT,
)
from app.router import RouteDecision, route_query_with_reason
from app.state import AgentState, AnalysisResult, ToolTraceItem
from app.tools import (
    count_rows_impl,
    filter_rows_impl,
    get_dataset_schema_impl,
    group_counts_impl,
    resolve_filter_value_impl,
    sample_examples_impl,
    summarize_rows_impl,
)


PlannerToolName = Literal[
    "get_dataset_schema",
    "resolve_filter_value",
    "filter_rows",
    "count_rows",
    "sample_examples",
    "group_counts",
    "summarize_rows",
    "read_user_profile",
]

VALID_PLANNER_TOOL_NAMES = {
    "get_dataset_schema",
    "resolve_filter_value",
    "filter_rows",
    "count_rows",
    "sample_examples",
    "group_counts",
    "summarize_rows",
    "read_user_profile",
}


class ProfileObservationDecision(BaseModel):
    """Decision about whether a durable profile observation should be saved."""

    observation: str = Field(
        default="",
        description="Concise durable observation to save, or empty string.",
    )


class ToolPlanDecision(BaseModel):
    """Planner decision for the next data-agent action."""

    action: Literal["call_tool", "final_answer"] = Field(
        description="Whether to call one tool or produce a final answer."
    )
    tool_name: PlannerToolName | Literal[""] = Field(
        default="",
        description=(
            "Tool to call when action is 'call_tool'. Must be empty when "
            "action is 'final_answer'."
        ),
    )
    tool_input: dict[str, Any] = Field(
        default_factory=dict,
        description="JSON-serializable input for the selected tool.",
    )
    final_answer: str = Field(
        default="",
        description="Final answer when no more tools are needed.",
    )
    reason: str = Field(
        description="Brief explanation of the planning decision."
    )


class ObservationReviewDecision(BaseModel):
    """Reviewer decision about whether observations answer the user."""

    status: Literal["answered", "needs_more", "cannot_answer"] = Field(
        description=(
            "answered if the trace is sufficient. "
            "needs_more only if one specific new tool call can add missing evidence. "
            "cannot_answer if the requested value/subset does not exist or no tool "
            "can add useful evidence."
        )
    )
    reason: str = Field(
        description=(
            "One short sentence explaining what the observations prove or miss. "
            "Do not include step-by-step reasoning."
        )
    )
    final_answer: str = Field(
        default="",
        description=(
            "Concise grounded final answer when status is answered or cannot_answer. "
            "Leave empty when status is needs_more."
        ),
    )
    suggested_tool_name: str = Field(
        default="",
        description=(
            "Required only when status is needs_more. Must be a new useful tool call, "
            "not a repeat of an already observed call. Empty otherwise."
        )
    )
    suggested_tool_input: dict[str, Any] = Field(
        default_factory=dict,
        description="Minimal next tool input when status is needs_more; otherwise empty.",
    )


@lru_cache(maxsize=1)
def get_agent_llm():
    """Return a cached OpenAI-compatible chat model for the data agent."""
    if not settings.nebius_api_key:
        raise RuntimeError(
            "NEBIUS_API_KEY is missing. Add it to your environment or .env file "
            "before using the graph agent."
        )

    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The graph agent requires 'langchain-openai'. "
            "Install project dependencies before running the agent."
        ) from exc

    return ChatOpenAI(
        model=settings.agent_model,
        api_key=settings.nebius_api_key,
        base_url=settings.nebius_base_url,
        temperature=0,
        max_tokens=settings.max_tokens,
        # extra_body={
        #     "enable_thinking": False,
        #     "thinking_budget": 0,
        # },
    )


@lru_cache(maxsize=1)
def get_structured_tool_planner_llm():
    """Return a cached model configured for next-tool planning decisions."""
    return get_agent_llm().with_structured_output(ToolPlanDecision)


@lru_cache(maxsize=1)
def get_structured_observation_reviewer_llm():
    """Return a cached model configured for observation-readiness decisions."""
    return get_agent_llm().with_structured_output(ObservationReviewDecision)


@lru_cache(maxsize=1)
def get_structured_profile_llm():
    """Return a cached model configured for profile-update decisions."""
    return get_agent_llm().with_structured_output(ProfileObservationDecision)


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


def _latest_user_message(messages: list[BaseMessage]) -> str:
    """Return the latest human message content from graph state."""
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return str(message.content)

    return ""


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


def _structured_results_for_prompt(results: list[AnalysisResult]) -> str:
    """Format recent structured results for follow-up resolution."""
    if not results:
        return "No recent structured results."

    lines: list[str] = []
    for index, result in enumerate(results[-5:], start=1):
        row_ids = result.get("row_ids")
        lines.append(
            f"{index}. label={result['label']}; "
            f"value={result['value']}; "
            f"query_type={result['query_type']}; "
            f"row_ids={summarize_row_ids(row_ids)}"
        )

    return "\n".join(lines)


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
) -> tuple[list[int], int] | None:
    """Return previous sample row IDs and next offset for 'show more' follow-ups."""
    for result in reversed(results):
        if result.get("query_type") != "sample":
            continue

        row_ids = result.get("row_ids")
        offset = result.get("value")

        if isinstance(row_ids, list) and isinstance(offset, int):
            return row_ids, offset

    return None


def _format_sample_examples_observation(
    row_ids: list[int] | None,
    n: int,
    offset: int,
) -> tuple[str, int]:
    """Call sample_examples and format the observation text."""
    result = sample_examples_impl(row_ids=row_ids, n=n, offset=offset)

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
        f"Returned {len(result.examples)} examples. "
        f"Next offset = {result.next_offset}. "
        + ("\n\n" + "\n\n---\n\n".join(example_lines) if example_lines else "")
    )

    return observation, result.next_offset


def _handle_more_examples_follow_up(state: AgentState) -> dict[str, Any] | None:
    """Deterministically answer 'show more examples' without asking the LLM."""
    user_query = _latest_user_message(state["messages"])

    if not _is_more_examples_query(user_query):
        return None

    sample_context = _latest_sample_context(state["last_structured_results"])
    if sample_context is None:
        return None

    row_ids, offset = sample_context
    n = _requested_example_count(user_query)

    observation, next_offset = _format_sample_examples_observation(
        row_ids=row_ids,
        n=n,
        offset=offset,
    )

    tool_input = {
        "row_ids": row_ids,
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
            row_ids=row_ids,
        ),
    )

    state["final_answer"] = observation

    return {
        "tool_trace": state["tool_trace"],
        "last_structured_results": state["last_structured_results"],
        "final_answer": observation,
        "messages": [AIMessage(content=observation)],
    }


def _compact_tool_input_for_prompt(tool_input: dict[str, Any]) -> dict[str, Any]:
    """Return tool input without exposing long row_id lists to the LLM."""
    compacted = dict(tool_input)
    row_ids = compacted.get("row_ids")

    if isinstance(row_ids, list):
        compacted["row_ids_summary"] = summarize_row_ids(row_ids)
        compacted["row_ids_access_note"] = (
            "Do not copy preview IDs. Use scope='latest_filter' when operating "
            "on the latest filtered subset."
        )
        del compacted["row_ids"]

    return compacted


def _compact_tool_trace_for_prompt(tool_trace: list[ToolTraceItem]) -> str:
    """Format the current turn trace for planner/reviewer prompts."""
    if not tool_trace:
        return "No tool calls yet in this turn."

    lines: list[str] = []
    for index, item in enumerate(tool_trace, start=1):
        tool_input = _compact_tool_input_for_prompt(item["tool_input"])
        lines.append(
            f"{index}. tool={item['tool_name']}\n"
             f"input={json.dumps(tool_input, ensure_ascii=False, default=str)}\n"
            f"observation={item['observation']}"
        )

    return "\n\n".join(lines)


def _profile_context_for_planner(state: AgentState) -> str:
    """Return profile context only when the user explicitly asks about memory.

    The dataset planner should not see the full durable user profile by default:
    task-specific profile pollution can bias tool planning. Profile content is
    still available through read_user_profile when the user asks profile/memory
    questions.
    """
    user_query = _latest_user_message(state["messages"]).lower()
    if "remember" in user_query or "profile" in user_query:
        return state["user_profile"]
    return "Profile hidden for dataset tool planning. Use read_user_profile only for explicit profile/memory questions."


def _build_planner_messages(
        state: AgentState,
        reviewer_feedback: str | None,
) -> list[BaseMessage]:
    """Build input messages for next-tool planning."""
    user_query = _latest_user_message(state["messages"])

    context = f"""Current route: {state["route"]}
Route reason: {state["route_reason"]}

User profile:
{_profile_context_for_planner(state)}

Recent structured results:
{_structured_results_for_prompt(state["last_structured_results"])}

Current turn tool trace:
{_compact_tool_trace_for_prompt(state["tool_trace"])}


Reviewer feedback:
{reviewer_feedback or "No reviewer feedback yet."}
"""

    return [
        SystemMessage(content=PLANNER_SYSTEM_PROMPT),
        SystemMessage(content=context),
        HumanMessage(content=user_query),
    ]


def _build_reviewer_messages(state: AgentState) -> list[BaseMessage]:
    """Build input messages for observation review."""
    user_query = _latest_user_message(state["messages"])
    context = f"""Current route: {state["route"]}
Route reason: {state["route_reason"]}

Recent structured results:
{_structured_results_for_prompt(state["last_structured_results"])}

Current turn tool trace:
{_compact_tool_trace_for_prompt(state["tool_trace"])}

"""

    return [
        SystemMessage(content=REVIEWER_SYSTEM_PROMPT),
        SystemMessage(content=context),
        HumanMessage(content=user_query),
    ]


def _normalize_tool_input(tool_input: dict[str, Any]) -> dict[str, Any]:
    """Return a shallow JSON-like copy of planner-supplied tool input."""
    return dict(tool_input or {})


def _tool_call_already_exists(
    state: AgentState,
    tool_name: str,
    tool_input: dict[str, Any],
) -> bool:
    """Return True when the same tool call already exists in this turn trace.
    A call with scope='latest_filter' is only a duplicate if it happened after
    the latest current-turn filter_rows call. The same raw scope can point to a
    different subset after a new filter_rows call.
    """
    normalized_input = _normalize_tool_input(tool_input)
    latest_filter_index = _latest_current_turn_filter_index(state)
    uses_latest_filter_scope = normalized_input.get("scope") in {
        "latest_filter",
        "latest_filtered_subset",
    }

    for index, item in enumerate(state["tool_trace"]):
        if item["tool_name"] != tool_name:
            continue
        if _normalize_tool_input(item["tool_input"]) != normalized_input:
            continue
        if (
            uses_latest_filter_scope
            and latest_filter_index is not None
            and index < latest_filter_index
        ):
            continue
        return True

    return False


def _format_model_dict(data: dict[str, Any]) -> str:
    """Format tool output as stable JSON for trace observations."""
    return json.dumps(data, ensure_ascii=False, default=str)


def _format_filter_rows_observation(result) -> str:
    """Format filter_rows output without dumping every matching row_id.

    Full row IDs are preserved in last_structured_results for follow-up logic.
    The observation only needs the exact match count, compact row-id summary,
    and applied filters.
    """
    return _format_model_dict(
        {
            "match_count": result.match_count,
            "row_ids_summary": summarize_row_ids(result.row_ids),
            "row_ids_access_note": (
                "Full row IDs are stored in graph state. Use "
                "scope='latest_filter' for the latest filtered subset."
            ),
            "applied_filters": result.applied_filters,
        }
    )


def _coerce_row_ids(value: Any) -> list[int] | None:
    """Coerce planner-supplied row IDs into integers or reject invalid values."""
    if value is None:
        return None

    if not isinstance(value, list):
        raise ValueError(
            "row_ids must be a list of integer row IDs or null; "
            f"got {type(value).__name__}."
        )

    row_ids: list[int] = []
    for item in value:
        try:
            row_ids.append(int(item))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "row_ids must contain only integer-compatible values; "
                f"got {item!r}."
            ) from exc

    return row_ids


def _resolve_row_ids_from_tool_input(
    state: AgentState,
    normalized_input: dict[str, Any],
) -> list[int] | None:
    """Resolve row_ids or the controlled latest-filter scope reference."""
    scope = normalized_input.get("scope")
    if scope in {"latest_filter", "latest_filtered_subset"}:
        row_ids = _latest_filter_row_ids(state)
        if row_ids is None:
            raise ValueError(
                "scope='latest_filter' was requested, but no filtered subset "
                "exists in recent structured results."
            )
        normalized_input["row_ids"] = row_ids
        return row_ids

    return _coerce_row_ids(normalized_input.get("row_ids"))


def _latest_filter_row_ids(state: AgentState) -> list[int] | None:
    """Return the latest full row_id subset produced by filter_rows."""
    for result in reversed(state["last_structured_results"]):
        if result.get("query_type") != "filter":
            continue

        row_ids = result.get("row_ids")
        if isinstance(row_ids, list):
            return row_ids

    return None


def _latest_current_turn_filter_index(state: AgentState) -> int | None:
    """Return the latest non-error filter_rows index in the current turn trace."""
    for index in range(len(state["tool_trace"]) - 1, -1, -1):
        item = state["tool_trace"][index]
        if item["tool_name"] != "filter_rows":
            continue
        if _trace_observation_is_error(item["observation"]):
            continue
        return index

    return None


def _latest_current_turn_filter_row_ids(state: AgentState) -> list[int] | None:
    """Return latest-filter row IDs only after filter_rows ran this turn."""
    if _latest_current_turn_filter_index(state) is None:
        return None

    return _latest_filter_row_ids(state)


def _resolve_scoped_group_count_row_ids(
    state: AgentState,
    normalized_input: dict[str, Any],
    group_by: str,
) -> list[int] | None:
    """Resolve row IDs for group_counts while preventing stale latest_filter use."""
    user_query = _latest_user_message(state["messages"])
    needs_filtered_scope = _requires_grouped_filtered_scope(user_query, group_by)
    scope = normalized_input.get("scope")

    if scope in {"latest_filter", "latest_filtered_subset"}:
        if needs_filtered_scope:
            row_ids = _latest_current_turn_filter_row_ids(state)
            if row_ids is None:
                raise ValueError(
                    "scope='latest_filter' was requested for a scoped distribution, "
                    "but filter_rows has not created a filtered subset in the "
                    "current turn."
                )
        else:
            row_ids = _latest_filter_row_ids(state)
            if row_ids is None:
                raise ValueError(
                    "scope='latest_filter' was requested, but no filtered subset "
                    "exists in recent structured results."
                )

        normalized_input["row_ids"] = row_ids
        return row_ids

    return _coerce_row_ids(normalized_input.get("row_ids"))


def _execute_selected_tool(
    state: AgentState,
    tool_name: str,
    tool_input: dict[str, Any],
) -> None:
    """Execute one selected tool and append trace/structured follow-up state."""
    normalized_input = _normalize_tool_input(tool_input)

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
            query=str(normalized_input.get("query", "")),
            columns=normalized_input.get("columns") or ["category", "intent"],
            top_k=int(normalized_input.get("top_k", 5)),
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
                row_ids=None,
            ),
        )
        return

    if tool_name == "filter_rows":
        if (
            normalized_input.get("limit") is not None
            and _is_example_request(_latest_user_message(state["messages"]))
        ):
            normalized_input = {
                **normalized_input,
                "limit": None,
            }

        result = filter_rows_impl(
            category=normalized_input.get("category"),
            intent=normalized_input.get("intent"),
            text_query=normalized_input.get("text_query"),
            limit=normalized_input.get("limit"),
        )
        _append_trace(
            state,
            tool_name,
            normalized_input,
            _format_filter_rows_observation(result),
        )
        _append_structured_result(
            state,
            AnalysisResult(
                label=str(result.applied_filters),
                value=result.match_count,
                query_type="filter",
                row_ids=result.row_ids,
            ),
        )
        return

    if tool_name == "count_rows":
        try:
            row_ids = _resolve_scoped_group_count_row_ids(
                state=state,
                normalized_input=normalized_input,
                group_by=group_by,
            )
        except ValueError as exc:
            _append_tool_error(
                state=state,
                tool_name=tool_name,
                tool_input=normalized_input,
                error=str(exc),
                required_next_step=(
                    "Use actual row IDs from filter_rows, scope='latest_filter', "
                    "or omit row_ids to count all rows."
                ),
            )
            return

        result = count_rows_impl(row_ids=row_ids)
        _append_trace(
            state,
            tool_name,
            normalized_input,
            _format_model_dict(result.model_dump()),
        )
        _append_structured_result(
            state,
            AnalysisResult(
                label="count_rows",
                value=result.count,
                query_type="count",
                row_ids=row_ids,
            ),
        )
        return

    if tool_name == "sample_examples":
        try:
            row_ids = _resolve_row_ids_from_tool_input(state, normalized_input)
        except ValueError:
            if (
                normalized_input.get("row_ids") is not None
                and _is_example_request(_latest_user_message(state["messages"]))
            ):
                latest_filter_row_ids = _latest_filter_row_ids(state)
                if latest_filter_row_ids is not None:
                    row_ids = latest_filter_row_ids
                    normalized_input["row_ids"] = row_ids
                else:
                    _append_tool_error(
                        state=state,
                        tool_name=tool_name,
                        tool_input=normalized_input,
                        error=(
                            "sample_examples received invalid row_ids and no "
                            "filtered subset exists to repair them."
                        ),
                        required_next_step=(
                            "Call filter_rows first, then sample_examples with "
                            "scope='latest_filter' or actual row IDs."
                        ),
                    )
                    return
            else:
                _append_tool_error(
                    state=state,
                    tool_name=tool_name,
                    tool_input=normalized_input,
                    error="sample_examples received invalid row_ids.",
                    required_next_step=(
                        "Use actual row IDs from filter_rows, scope='latest_filter', "
                        "or omit row_ids for all rows."
                    ),
                )
                return

        n = int(normalized_input.get("n", 3))
        offset = int(normalized_input.get("offset", 0))
        observation, next_offset = _format_sample_examples_observation(
            row_ids=row_ids,
            n=n,
            offset=offset,
        )
        _append_trace(state, tool_name, normalized_input, observation)
        _append_structured_result(
            state,
            AnalysisResult(
                label="sample_examples",
                value=next_offset,
                query_type="sample",
                row_ids=row_ids,
            ),
        )
        return

    if tool_name == "group_counts":
        group_by = normalized_input.get("group_by")
        if group_by not in {"category", "intent"}:
            raise ValueError(
                "group_counts requires group_by='category' or group_by='intent'."
            )

        try:
            row_ids = _resolve_row_ids_from_tool_input(state, normalized_input)
        except ValueError as exc:
            _append_tool_error(
                state=state,
                tool_name=tool_name,
                tool_input=normalized_input,
                error=str(exc),
                required_next_step=(
                    "For filtered distributions, call filter_rows first in the "
                    "current turn and then call group_counts with "
                    "scope='latest_filter'."
                ),
            )
            return

        user_query = _latest_user_message(state["messages"])
        if row_ids is None and _requires_grouped_filtered_scope(user_query, group_by):
            _append_tool_error(
                state=state,
                tool_name=tool_name,
                tool_input=normalized_input,
                error=(
                    "group_counts would group all rows, but the user asked for a "
                    "distribution inside a filtered category or intent."
                ),
                required_next_step=(
                    "Resolve the requested value, call filter_rows, then call "
                    "group_counts with scope='latest_filter' or actual row IDs."
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

        result = group_counts_impl(
            group_by=group_by,
            row_ids=row_ids,
            top_k=top_k,
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
                label=f"group_counts:{group_by}",
                value=len(result.counts),
                query_type="group_counts",
                row_ids=row_ids,
            ),
        )
        return

    if tool_name == "summarize_rows":
        try:
            row_ids = _resolve_row_ids_from_tool_input(state, normalized_input)
        except ValueError as exc:
            _append_tool_error(
                state=state,
                tool_name=tool_name,
                tool_input=normalized_input,
                error=str(exc),
                required_next_step=(
                    "Call filter_rows first, then summarize_rows with "
                    "scope='latest_filter' or actual row IDs."
                ),
            )
            return

        row_ids = row_ids or []
        result = summarize_rows_impl(
            row_ids=row_ids,
            focus=str(
                normalized_input.get("focus", _latest_user_message(state["messages"]))
            ),
            max_examples=int(normalized_input.get("max_examples", 100)),
        )
        _append_trace(
            state,
            tool_name,
            normalized_input,
            _format_model_dict(result.model_dump()),
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


def _has_prior_filter_trace(
    state: AgentState,
    max_index: int,
    required_filter_column: str,
) -> bool:
    """Return True when a filter_rows call for the required column precedes a tool."""
    for item in state["tool_trace"][:max_index]:
        if item["tool_name"] != "filter_rows":
            continue
        if item["tool_input"].get(required_filter_column):
            return True
    return False


def _has_valid_scoped_group_counts_trace(
    state: AgentState,
    group_by: str,
    required_filter_column: str,
) -> bool:
    """Validate scoped distribution evidence using the existing tool trace only."""
    for index, item in enumerate(state["tool_trace"]):
        if item["tool_name"] != "group_counts":
            continue
        if item["tool_input"].get("group_by") != group_by:
            continue
        if _trace_observation_is_error(item["observation"]):
            continue

        row_ids = item["tool_input"].get("row_ids")
        if not isinstance(row_ids, list):
            continue

        if _has_prior_filter_trace(
            state=state,
            max_index=index,
            required_filter_column=required_filter_column,
        ):
            return True

    return False


def _has_filter_trace_for_column(state: AgentState, column: str) -> bool:
    """Return True when this turn already has a filter_rows call for column."""
    return any(
        item["tool_name"] == "filter_rows"
        and bool(item["tool_input"].get(column))
        and not _trace_observation_is_error(item["observation"])
        for item in state["tool_trace"]
    )


def _has_resolver_trace_for_column(state: AgentState, column: str) -> bool:
    """Return True when this turn has a resolver call that searched column."""
    for item in state["tool_trace"]:
        if item["tool_name"] != "resolve_filter_value":
            continue
        columns = item["tool_input"].get("columns") or []
        if column in columns and not _trace_observation_is_error(item["observation"]):
            return True
    return False


def _answer_contract_error(state: AgentState) -> str | None:
    """Block final answers for scoped distributions until the scoped group exists."""
    user_query = _latest_user_message(state["messages"])

    if _requires_grouped_filtered_scope(user_query, group_by="intent"):
        if _has_valid_scoped_group_counts_trace(
            state=state,
            group_by="intent",
            required_filter_column="category",
        ):
            return None

        if _has_filter_trace_for_column(state, "category"):
            return (
                "The user asked for an intent distribution inside a category. "
                "filter_rows has created the category subset, but there is no "
                "valid group_counts(group_by='intent') over that subset yet."
            )

        if _has_resolver_trace_for_column(state, "category"):
            return (
                "The user asked for an intent distribution inside a category. "
                "The category has been resolved, but filter_rows has not created "
                "the category subset yet."
            )

        return (
            "The user asked for an intent distribution inside a category. "
            "Valid evidence requires resolving the category, filtering rows to "
            "that category, and grouping intents over that filtered subset."
        )

    if _requires_grouped_filtered_scope(user_query, group_by="category"):
        if _has_valid_scoped_group_counts_trace(
            state=state,
            group_by="category",
            required_filter_column="intent",
        ):
            return None

        if _has_filter_trace_for_column(state, "intent"):
            return (
                "The user asked for a category distribution inside an intent. "
                "filter_rows has created the intent subset, but there is no "
                "valid group_counts(group_by='category') over that subset yet."
            )

        if _has_resolver_trace_for_column(state, "intent"):
            return (
                "The user asked for a category distribution inside an intent. "
                "The intent has been resolved, but filter_rows has not created "
                "the intent subset yet."
            )

        return (
            "The user asked for a category distribution inside an intent. "
            "Valid evidence requires resolving the intent, filtering rows to "
            "that intent, and grouping categories over that filtered subset."
        )

    return None


def _build_final_answer_block_feedback(contract_error: str) -> str:
    """Convert deterministic contract validation into planner feedback."""
    return (
        f"Final answer blocked by deterministic validation: {contract_error} "
        "Do not answer from the current observations. Call the missing tool. "
        "For scoped grouping, use actual row IDs from filter_rows or "
        "scope='latest_filter'."
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
                return {
                    "tool_trace": state["tool_trace"],
                                "last_structured_results": state["last_structured_results"],
                    "final_answer": final_answer,
                    "messages": [AIMessage(content=final_answer)],
                }

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

            if (
                    plan.tool_name == "sample_examples"
                    and _is_example_request(_latest_user_message(state["messages"]))
            ):
                final_answer = state["tool_trace"][-1]["observation"]
                state["final_answer"] = final_answer
                _debug_trace(
                    f"iteration {iteration_number}/{state['max_iterations']}: "
                    "returning deterministic sample_examples answer"
                )
                return {
                    "tool_trace": state["tool_trace"],
                                "last_structured_results": state["last_structured_results"],
                    "final_answer": final_answer,
                    "messages": [AIMessage(content=final_answer)],
                }
            _debug_trace(
                f"iteration {iteration_number}/{state['max_iterations']}: "
                "reviewer start"
            )
            try:
                reviewer_llm = get_structured_observation_reviewer_llm()
                review = reviewer_llm.invoke(_build_reviewer_messages(state))
                if not isinstance(review, ObservationReviewDecision):
                    review = ObservationReviewDecision.model_validate(review)
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
                state["final_answer"] = final_answer
                _debug_trace(
                    f"iteration {iteration_number}/{state['max_iterations']}: "
                    f"returning reviewer status={review.status}"
                )
                return {
                    "tool_trace": state["tool_trace"],
                                "last_structured_results": state["last_structured_results"],
                    "final_answer": final_answer,
                    "messages": [AIMessage(content=final_answer)],
                }

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

                reviewer_feedback = (
                    f"{review.reason}\n"
                    f"Suggested next tool: {review.suggested_tool_name}\n"
                    f"Suggested next input: "
                    f"{json.dumps(review.suggested_tool_input, ensure_ascii=False, default=str)}"
                )
                _debug_trace(
                    f"iteration {iteration_number}/{state['max_iterations']}: "
                    f"reviewer requested next tool={review.suggested_tool_name}"
                )

        except Exception as exc:
            fallback = _fallback_answer()
            state["final_answer"] = fallback
            _debug_trace(
                f"iteration {iteration_number}/{state['max_iterations']}: "
                f"exception={type(exc).__name__}: {exc}"
            )
            return {
                "tool_trace": state["tool_trace"],
                        "last_structured_results": state["last_structured_results"],
                "final_answer": fallback,
                "messages": [AIMessage(content=fallback)],
            }

    fallback = _fallback_answer()
    state["final_answer"] = fallback
    _debug_trace(
        f"max_iterations_exhausted={state['max_iterations']}; "
        f"trace_steps={len(state['tool_trace'])}"
    )

    return {
        "tool_trace": state["tool_trace"],
        "last_structured_results": state["last_structured_results"],
        "final_answer": fallback,
        "messages": [AIMessage(content=fallback)],
    }


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