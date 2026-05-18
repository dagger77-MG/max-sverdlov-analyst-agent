from __future__ import annotations

from typing import TypedDict

from langchain_core.messages import BaseMessage

from app.router import RouteType


class ToolTraceItem(TypedDict):
    """One visible reasoning/tool step for CLI and Streamlit display."""

    tool_name: str
    tool_input: dict
    observation: str


class AnalysisResult(TypedDict):
    """A compact stored result used for follow-up questions."""

    label: str
    value: int | float | str
    query_type: str
    row_ids: list[int] | None


class AgentState(TypedDict):
    """LangGraph state shared across graph nodes."""

    messages: list[BaseMessage]
    session_id: str
    user_id: str
    route: RouteType | None
    route_reason: str | None
    tool_trace: list[ToolTraceItem]
    last_structured_results: list[AnalysisResult]
    user_profile: str
    iteration_count: int
    final_answer: str | None