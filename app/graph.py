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
            "answered if the trace is sufficient, needs_more if another tool call "
            "is required, or cannot_answer if the tools cannot answer the request."
        )
    )
    reason: str = Field(
        description="Brief explanation of what the observations prove or miss."
    )
    final_answer: str = Field(
        default="",
        description="Grounded final answer when status is answered or cannot_answer.",
    )
    suggested_tool_name: str = Field(
        default="",
        description="Recommended next tool when status is needs_more.",
    )
    suggested_tool_input: dict[str, Any] = Field(
        default_factory=dict,
        description="Recommended next tool input when status is needs_more.",
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
        max_tokens=2056,
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


def _compact_tool_trace_for_prompt(tool_trace: list[ToolTraceItem]) -> str:
    """Format the current turn trace for planner/reviewer prompts."""
    if not tool_trace:
        return "No tool calls yet in this turn."

    lines: list[str] = []
    for index, item in enumerate(tool_trace, start=1):
        lines.append(
            f"{index}. tool={item['tool_name']}\n"
            f"input={json.dumps(item['tool_input'], ensure_ascii=False, default=str)}\n"
            f"observation={item['observation']}"
        )

    return "\n\n".join(lines)


def _build_planner_messages(
        state: AgentState,
        reviewer_feedback: str | None,
) -> list[BaseMessage]:
    """Build input messages for next-tool planning."""
    user_query = _latest_user_message(state["messages"])

    context = f"""Current route: {state["route"]}
Route reason: {state["route_reason"]}

User profile:
{state["user_profile"]}

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
            "row_ids": summarize_row_ids(result.row_ids),
            "applied_filters": result.applied_filters,
        }
    )


def _coerce_row_ids(value: Any) -> list[int] | None:
    """Coerce planner-supplied row IDs into integers when possible."""
    if value is None:
        return None

    if not isinstance(value, list):
        return None

    row_ids: list[int] = []
    for item in value:
        try:
            row_ids.append(int(item))
        except (TypeError, ValueError):
            return None

    return row_ids


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
        row_ids = _coerce_row_ids(normalized_input.get("row_ids"))
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
        row_ids = _coerce_row_ids(normalized_input.get("row_ids"))
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
        row_ids = _coerce_row_ids(normalized_input.get("row_ids"))
        top_k = int(normalized_input.get("top_k", 20))
        result = group_counts_impl(
            group_by=group_by,
            row_ids=row_ids,
            top_k=top_k,
        )
        result_dict = result.model_dump()
        _append_trace(
            state,
            tool_name,
            normalized_input,
            _format_model_dict(result_dict),
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
        row_ids = _coerce_row_ids(normalized_input.get("row_ids")) or []
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


def _fallback_answer() -> str:
    """Return a safe graph fallback for planner/reviewer/tool errors."""
    return (
        "I could not complete the analysis within the allowed number of "
        "reasoning steps. Please try asking a more specific dataset question."
    )


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

    for _ in range(state["max_iterations"]):
        try:
            planner_llm = get_structured_tool_planner_llm()
            plan = planner_llm.invoke(
                _build_planner_messages(state, reviewer_feedback)
            )
            if not isinstance(plan, ToolPlanDecision):
                plan = ToolPlanDecision.model_validate(plan)

            if plan.action == "final_answer":
                final_answer = plan.final_answer.strip()
                if not final_answer:
                    reviewer_feedback = (
                        "Planner chose final_answer but returned empty text."
                    )
                    continue
                state["final_answer"] = final_answer
                return {
                    "tool_trace": state["tool_trace"],
                    "last_structured_results": state["last_structured_results"],
                    "final_answer": final_answer,
                    "messages": [AIMessage(content=final_answer)],
                }

            _execute_selected_tool(
                state=state,
                tool_name=plan.tool_name,
                tool_input=plan.tool_input,
            )

            if (
                    plan.tool_name == "sample_examples"
                    and _is_example_request(_latest_user_message(state["messages"]))
            ):
                final_answer = state["tool_trace"][-1]["observation"]
                state["final_answer"] = final_answer
                return {
                    "tool_trace": state["tool_trace"],
                    "last_structured_results": state["last_structured_results"],
                    "final_answer": final_answer,
                    "messages": [AIMessage(content=final_answer)],
                }

            reviewer_llm = get_structured_observation_reviewer_llm()
            review = reviewer_llm.invoke(_build_reviewer_messages(state))
            if not isinstance(review, ObservationReviewDecision):
                review = ObservationReviewDecision.model_validate(review)

            reviewer_feedback = review.reason

            if review.status in {"answered", "cannot_answer"}:
                final_answer = review.final_answer.strip() or review.reason.strip()
                state["final_answer"] = final_answer
                return {
                    "tool_trace": state["tool_trace"],
                    "last_structured_results": state["last_structured_results"],
                    "final_answer": final_answer,
                    "messages": [AIMessage(content=final_answer)],
                }

            if review.status == "needs_more" and review.suggested_tool_name:
                reviewer_feedback = (
                    f"{review.reason}\n"
                    f"Suggested next tool: {review.suggested_tool_name}\n"
                    f"Suggested next input: "
                    f"{json.dumps(review.suggested_tool_input, ensure_ascii=False, default=str)}"
                )

        except Exception:
            fallback = _fallback_answer()
            state["final_answer"] = fallback
            return {
                "tool_trace": state["tool_trace"],
                "last_structured_results": state["last_structured_results"],
                "final_answer": fallback,
                "messages": [AIMessage(content=fallback)],
            }

    fallback = _fallback_answer()
    state["final_answer"] = fallback

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