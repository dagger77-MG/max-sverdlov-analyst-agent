from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from app.prompts import PLANNER_SYSTEM_PROMPT, REVIEWER_SYSTEM_PROMPT
from app.state import AgentState, AnalysisResult, ToolTraceItem


def latest_user_message(messages: list[BaseMessage]) -> str:
    """Return the latest human message content from graph state."""
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return str(message.content)

    return ""


def _structured_results_for_prompt(results: list[AnalysisResult]) -> str:
    """Format recent structured results for follow-up resolution."""
    if not results:
        return "No recent structured results."

    lines: list[str] = []
    for index, result in enumerate(results[-5:], start=1):
        filters = result.get("filters") or {}
        lines.append(
            f"{index}. label={result['label']}; "
            f"value={result['value']}; "
            f"query_type={result['query_type']}; "
            f"filters={json.dumps(filters, ensure_ascii=False, default=str)}"
        )

    return "\n".join(lines)


def _compact_tool_input_for_prompt(tool_input: dict[str, Any]) -> dict[str, Any]:
    """Return a prompt-safe copy of a tool input dictionary."""
    return dict(tool_input)


def _compact_tool_trace_for_prompt(tool_trace: list[ToolTraceItem]) -> str:
    """Format the current turn trace for planner/reviewer prompts."""
    if not tool_trace:
        return "No tool calls yet in this turn."

    lines: list[str] = []
    tool_items = [
        item for item in tool_trace
        if item.get("event_type", "tool") == "tool"
    ]
    for index, item in enumerate(tool_items, start=1):
        tool_input = _compact_tool_input_for_prompt(item["tool_input"])
        lines.append(
            f"{index}. tool={item['tool_name']}\n"
            f"input={json.dumps(tool_input, ensure_ascii=False, default=str)}\n"
            f"observation={item['observation']}"
        )

    return "\n\n".join(lines) if lines else "No tool calls yet in this turn."


def _profile_context_for_planner(state: AgentState) -> str:
    """Return profile context only when the user explicitly asks about memory.

    The dataset planner should not see the full durable user profile by default:
    task-specific profile pollution can bias tool planning. Profile content is
    still available through read_user_profile when the user asks profile/memory
    questions.
    """
    user_query = latest_user_message(state["messages"]).lower()
    if "remember" in user_query or "profile" in user_query:
        return state["user_profile"]
    return (
        "Profile hidden for dataset tool planning. Use read_user_profile only "
        "for explicit profile/memory questions."
    )


def _build_planner_messages(
    state: AgentState,
    reviewer_feedback: str | None,
) -> list[BaseMessage]:
    """Build input messages for next-tool planning."""
    user_query = latest_user_message(state["messages"])

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
    user_query = latest_user_message(state["messages"])
    context = f"""Current route: {state["route"]}
Route reason: {state["route_reason"]}

Current turn tool trace:
{_compact_tool_trace_for_prompt(state["tool_trace"])}

Evidence boundary: judge only the current turn tool trace. Previous structured
results are planner context for follow-up interpretation, not current-turn evidence.
"""

    return [
        SystemMessage(content=REVIEWER_SYSTEM_PROMPT),
        SystemMessage(content=context),
        HumanMessage(content=user_query),
    ]