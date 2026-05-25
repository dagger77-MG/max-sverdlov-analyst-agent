from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


PlannerToolName = Literal[
    "get_dataset_schema",
    "resolve_filter_value",
    "count_rows",
    "sample_examples",
    "group_counts",
    "summarize_rows",
    "read_user_profile",
]

VALID_PLANNER_TOOL_NAMES = {
    "get_dataset_schema",
    "resolve_filter_value",
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
        max_length=512,
        description=(
            "One short sentence explaining the planning decision. "
            "Maximum 512 characters. Do not include hidden reasoning, "
            "chains of thought, self-debate, or long analysis."
        )
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