from __future__ import annotations

import json
from typing import Any

from app.state import ToolTraceItem


def compact_tool_input(tool_input: dict[str, Any], max_list_items: int = 10) -> dict[str, Any]:
    """Return a display-friendly copy of a tool input dictionary.

    Large row_id lists can make CLI and Streamlit traces unreadable, so this
    function summarizes long lists while keeping the trace useful.
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


def format_trace_item(trace_item: ToolTraceItem) -> str:
    """Format a single tool trace item for CLI display."""
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
    """Format route and tool trace information for human-readable logs."""
    lines: list[str] = []

    if route is not None:
        lines.append(f"[router] {route}")

    if route_reason:
        lines.append(f"[router_reason] {route_reason}")

    for trace_item in tool_trace:
        lines.append(format_trace_item(trace_item))

    return "\n\n".join(lines)


def summarize_row_ids(row_ids: list[int] | None) -> str:
    """Return a compact textual summary of row IDs for observations."""
    if row_ids is None:
        return "all rows"

    if not row_ids:
        return "0 row IDs"

    preview = ", ".join(str(row_id) for row_id in row_ids[:5])
    suffix = "..." if len(row_ids) > 10 else ""

    return f"{len(row_ids)} row IDs [{preview}{suffix}]"