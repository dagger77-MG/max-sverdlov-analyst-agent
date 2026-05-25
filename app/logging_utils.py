from __future__ import annotations

import json
from typing import Any

from app.state import ToolTraceItem


def compact_tool_input(
    tool_input: dict[str, Any],
    max_list_items: int = 10,
) -> dict[str, Any]:
    """Return a display-friendly copy of a tool input dictionary.

    Long lists can make CLI and Streamlit traces unreadable, so this function
    summarizes them while keeping the trace useful.
    """
    compacted: dict[str, Any] = {}

    for key, value in tool_input.items():
        if isinstance(value, list) and len(value) > max_list_items:
            compacted[key] = {
                "type": "list",
                "count": len(value),
                "preview": value[:max_list_items],
            }
        else:
            compacted[key] = value

    return compacted


def format_tool_input(tool_input: dict[str, Any]) -> str:
    """Format tool input as readable JSON."""
    return json.dumps(
        compact_tool_input(tool_input),
        ensure_ascii=False,
        indent=2,
        default=str,
    )


def format_reviewer_trace_item(trace_item: ToolTraceItem) -> str:
    """Format a reviewer decision trace item for CLI and Streamlit display."""
    status = trace_item.get("reviewer_status", "unknown")
    reason = trace_item.get("reviewer_reason", "")
    suggested_tool_name = trace_item.get("suggested_tool_name", "")
    suggested_tool_input = trace_item.get("suggested_tool_input", {})

    lines = [f"[reviewer] {status}"]

    if reason:
        lines.append(f"[reason] {reason}")

    if suggested_tool_name:
        lines.append(f"[suggested_tool] {suggested_tool_name}")

    if suggested_tool_input:
        lines.append(
            "[suggested_input]\n"
            f"{format_tool_input(suggested_tool_input)}"
        )

    return "\n".join(lines)


def format_trace_item(trace_item: ToolTraceItem) -> str:
    """Format a single reasoning trace item for CLI and Streamlit display."""
    if trace_item.get("event_type", "tool") == "reviewer":
        return format_reviewer_trace_item(trace_item)

    tool_name = trace_item["tool_name"]
    tool_input = format_tool_input(trace_item["tool_input"])
    observation = trace_item["observation"]

    return (
        f"[tool] {tool_name}\n"
        f"[input]\n{tool_input}\n"
        f"[observation] {observation}"
    )


def format_reasoning_trace(
    route: str | None,
    route_reason: str | None,
    tool_trace: list[ToolTraceItem],
) -> str:
    """Format route and reasoning trace information for human-readable logs."""
    lines: list[str] = []

    if route is not None:
        lines.append(f"[router] {route}")

    if route_reason:
        lines.append(f"[router_reason] {route_reason}")

    for trace_item in tool_trace:
        lines.append(format_trace_item(trace_item))

    return "\n\n".join(lines)