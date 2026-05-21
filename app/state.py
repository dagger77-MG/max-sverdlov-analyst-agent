from __future__ import annotations

from typing import Annotated, Any, NotRequired, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from app.router import RouteType


class DatasetFilters(TypedDict):
    """Semantic dataset filters stored for follow-up questions."""

    category: str | None
    intent: str | None
    text_query: str | None


class ToolTraceItem(TypedDict):
    """One visible reasoning/tool step for CLI and Streamlit display."""

    tool_name: str
    tool_input: dict[str, Any]
    observation: str


class AnalysisResult(TypedDict):
    """A compact stored result used for follow-up questions.
    Store semantic filters instead of row IDs so long row-id lists are never
    passed through graph state into planner/reviewer prompts.
    """

    label: str
    value: int | float | str
    query_type: str
    filters: NotRequired[DatasetFilters]
    match_count: NotRequired[int | None]


class AgentState(TypedDict):
    """LangGraph state shared across graph nodes."""

    messages: Annotated[list[BaseMessage], add_messages]
    session_id: str
    user_id: str
    route: RouteType | None
    route_reason: str | None
    tool_trace: list[ToolTraceItem]
    last_structured_results: list[AnalysisResult]
    user_profile: str
    max_iterations: int
    final_answer: str | None