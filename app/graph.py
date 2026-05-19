from __future__ import annotations

import ast
import json
import re
import sqlite3
from functools import lru_cache
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from app.config import settings
from app.langchain_tools import LANGCHAIN_TOOLS
from app.logging_utils import summarize_row_ids
from app.memory import read_user_profile_impl, update_user_profile_impl
from app.prompts import (
    DATA_AGENT_SYSTEM_PROMPT,
    OUT_OF_SCOPE_REFUSAL,
    PROFILE_UPDATE_SYSTEM_PROMPT,
)
from app.router import RouteDecision, route_query_with_reason
from app.state import AgentState, AnalysisResult, ToolTraceItem
from app.tools import sample_examples_impl


class ProfileObservationDecision(BaseModel):
    """Decision about whether a durable profile observation should be saved."""

    observation: str = Field(
        default="",
        description="Concise durable observation to save, or empty string.",
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
        extra_body={
            "enable_thinking": False,
            "thinking_budget": 0,
        },
    )


@lru_cache(maxsize=1)
def get_langchain_data_agent():
    """Return the standard LangChain data agent runtime."""
    try:
        from langchain.agents import create_agent
    except ImportError as exc:
        raise RuntimeError(
            "The standard LangChain agent requires 'langchain'. "
            "Install project dependencies before running the agent."
        ) from exc

    return create_agent(
        model=get_agent_llm(),
        tools=LANGCHAIN_TOOLS,
        system_prompt="/no_think\n" + DATA_AGENT_SYSTEM_PROMPT,
    )


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


def _message_content_as_text(message: BaseMessage) -> str:
    """Return message content as display-safe text."""
    content = getattr(message, "content", "")

    if isinstance(content, str):
        return content

    return str(content)


def _safe_parse_tool_output(content: Any) -> Any:
    """Best-effort parse of a LangChain tool message content."""
    if isinstance(content, dict | list):
        return content

    if not isinstance(content, str):
        return content

    stripped = content.strip()
    if not stripped:
        return stripped

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    try:
        return ast.literal_eval(stripped)
    except (SyntaxError, ValueError):
        return stripped


def _build_langchain_agent_messages(state: AgentState) -> list[BaseMessage]:
    """Build compact input messages for the standard LangChain agent."""
    user_query = _latest_user_message(state["messages"])

    context = f"""Current route: {state["route"]}
Route reason: {state["route_reason"]}

User profile:
{state["user_profile"]}

Recent structured results:
{_structured_results_for_prompt(state["last_structured_results"])}

Follow-up guidance:
- If the user asks for a total of recent counts, use recent structured results when available.
- Example-pagination follow-ups are handled deterministically by the graph when previous sample context is available.
- Do not reconstruct row IDs from compact previews.
- Do not answer from general knowledge.
"""

    return [
        SystemMessage(content=context),
        HumanMessage(content=user_query),
    ]


def _extract_langchain_tool_trace_and_results(
    messages: list[BaseMessage],
) -> tuple[list[ToolTraceItem], list[AnalysisResult]]:
    """Extract visible tool steps and compact follow-up results from agent messages."""
    trace: list[ToolTraceItem] = []
    structured_results: list[AnalysisResult] = []
    pending_tool_calls: dict[str, tuple[str, dict[str, Any]]] = {}

    for message in messages:
        tool_calls = getattr(message, "tool_calls", None) or []
        for tool_call in tool_calls:
            call_id = str(tool_call.get("id", ""))
            tool_name = str(tool_call.get("name", "unknown_tool"))
            tool_args = tool_call.get("args") or {}
            if not isinstance(tool_args, dict):
                tool_args = {"input": tool_args}
            pending_tool_calls[call_id] = (tool_name, tool_args)

        if getattr(message, "type", None) != "tool":
            continue

        tool_call_id = str(getattr(message, "tool_call_id", ""))
        tool_name, tool_input = pending_tool_calls.get(
            tool_call_id,
            (str(getattr(message, "name", "unknown_tool")), {}),
        )
        observation_text = _message_content_as_text(message)

        trace.append(
            ToolTraceItem(
                tool_name=tool_name,
                tool_input=tool_input,
                observation=observation_text,
            )
        )

        parsed_output = _safe_parse_tool_output(getattr(message, "content", ""))

        if tool_name == "filter_rows" and isinstance(parsed_output, dict):
            row_ids = parsed_output.get("row_ids")
            match_count = parsed_output.get("match_count")
            if isinstance(row_ids, list) and isinstance(match_count, int):
                structured_results.append(
                    AnalysisResult(
                        label=str(parsed_output.get("applied_filters", {})),
                        value=match_count,
                        query_type="filter",
                        row_ids=row_ids,
                    )
                )

        if tool_name == "count_rows" and isinstance(parsed_output, dict):
            count = parsed_output.get("count")
            if isinstance(count, int):
                row_ids = tool_input.get("row_ids")
                structured_results.append(
                    AnalysisResult(
                        label="count_rows",
                        value=count,
                        query_type="count",
                        row_ids=row_ids if isinstance(row_ids, list) else None,
                    )
                )

        if tool_name == "sample_examples" and isinstance(parsed_output, dict):
            next_offset = parsed_output.get("next_offset")
            row_ids = tool_input.get("row_ids")
            if isinstance(next_offset, int):
                structured_results.append(
                    AnalysisResult(
                        label="sample_examples",
                        value=next_offset,
                        query_type="sample",
                        row_ids=row_ids if isinstance(row_ids, list) else None,
                    )
                )

    return trace, structured_results


def _extract_final_answer(messages: list[BaseMessage]) -> str:
    """Extract the final AI answer from LangChain agent messages."""
    for message in reversed(messages):
        if not isinstance(message, AIMessage):
            continue

        if getattr(message, "tool_calls", None):
            continue

        content = _message_content_as_text(message).strip()
        if content:
            return content

    return "I completed the analysis, but no final answer was provided."


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


def langchain_data_agent_node(state: AgentState) -> dict[str, Any]:
    """Run the standard LangChain agent runtime for dataset questions."""
    deterministic_result = _handle_more_examples_follow_up(state)
    if deterministic_result is not None:
        return deterministic_result

    data_agent = get_langchain_data_agent()

    try:
        result = data_agent.invoke(
            {
                "messages": _build_langchain_agent_messages(state),
            },
            config={
                "recursion_limit": state["max_iterations"] + 5,
            },
        )
    except Exception:
        fallback = (
            "I could not complete the analysis within the allowed number of "
            "reasoning steps. Please try asking a more specific dataset question."
        )
        state["final_answer"] = fallback
        return {
            "tool_trace": state["tool_trace"],
            "last_structured_results": state["last_structured_results"],
            "final_answer": fallback,
            "messages": [AIMessage(content=fallback)],
        }

    result_messages = result.get("messages", [])
    if not isinstance(result_messages, list):
        result_messages = []

    tool_trace, new_structured_results = _extract_langchain_tool_trace_and_results(
        result_messages
    )

    state["tool_trace"] = tool_trace
    for structured_result in new_structured_results:
        _append_structured_result(state, structured_result)

    final_answer = _extract_final_answer(result_messages)
    state["final_answer"] = final_answer

    return {
        "tool_trace": state["tool_trace"],
        "last_structured_results": state["last_structured_results"],
        "final_answer": final_answer,
        "messages": [AIMessage(content=final_answer)],
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

    return "langchain_data_agent_node"


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
    graph_builder.add_node("langchain_data_agent_node", langchain_data_agent_node)
    graph_builder.add_node("refusal_node", refusal_node)
    graph_builder.add_node("profile_update_node", profile_update_node)

    graph_builder.add_edge(START, "load_user_profile_node")
    graph_builder.add_edge("load_user_profile_node", "router_node")
    graph_builder.add_conditional_edges(
        "router_node",
        route_after_router,
        {
            "langchain_data_agent_node": "langchain_data_agent_node",
            "refusal_node": "refusal_node",
        },
    )
    graph_builder.add_edge("langchain_data_agent_node", "profile_update_node")
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