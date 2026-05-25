from __future__ import annotations

from typing import Annotated, Any, Literal, NotRequired, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from app.router import RouteType


TraceEventType = Literal["tool", "reviewer"]
ReviewerTraceStatus = Literal["answered", "needs_more", "cannot_answer", "error"]


class DatasetFilters(TypedDict):
    """Semantic dataset filters stored for follow-up questions."""

    category: str | None
    intent: str | None
    text_query: str | None


class ToolTraceItem(TypedDict, total=False):
    """One visible reasoning event for CLI and Streamlit display."""

    event_type: TraceEventType

    # Tool event fields.
    tool_name: str
    tool_input: dict[str, Any]
    observation: str

    # Reviewer event fields.
    reviewer_status: ReviewerTraceStatus
    reviewer_reason: str
    reviewer_final_answer: str
    suggested_tool_name: str
    suggested_tool_input: dict[str, Any]


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