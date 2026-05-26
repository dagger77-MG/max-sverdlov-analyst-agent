from __future__ import annotations

import logging
import re
import sqlite3
from functools import lru_cache
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph

from app.agent import loop as _loop
from app.agent import profile as _profile
from app.agent.context import latest_user_message
from app.config import settings
from app.prompts import OUT_OF_SCOPE_REFUSAL
from app.router import RouteDecision, route_query_with_reason
from app.state import AgentState

logger = logging.getLogger(__name__)

CAPABILITIES_HELP_RESPONSE = (
    "I can analyze the Bitext Customer Service dataset: counts, examples, "
    "category/intent breakdowns, summaries, response patterns, follow-up "
    "questions, and your saved profile."
)


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


def _is_capabilities_query(query: str) -> bool:
    """Return True when the user asks what this agent can do."""
    normalized = re.sub(r"[^a-z0-9]+", " ", query.lower()).strip()

    if not normalized:
        return False

    exact_queries = {
        "help",
        "what can you do",
        "what do you do",
        "how can i use you",
        "how can i use this agent",
        "what questions can i ask",
        "what can i ask",
        "explain your capabilities",
        "show your capabilities",
        "what is your purpose",
        "what is this agent for",
        "what are you for",
    }

    if normalized in exact_queries:
        return True

    return bool(
        re.search(r"\bwhat\b.*\b(can|could)\b.*\b(agent|you)\b.*\bdo\b", normalized)
        or re.search(r"\bhow\b.*\b(use|work with)\b.*\b(agent|you)\b", normalized)
        or re.search(r"\b(available|supported)\b.*\bquestions\b", normalized)
        or re.search(r"\bcapabilities\b", normalized)
    )


def route_after_profile_load(state: AgentState) -> str:
    """Choose whether to answer capabilities/help before router classification."""
    if _is_capabilities_query(latest_user_message(state["messages"])):
        return "capabilities_help_node"

    return "router_node"


def capabilities_help_node(state: AgentState) -> dict[str, Any]:
    """Return a deterministic short capabilities answer without LLM routing."""
    return {
        "final_answer": CAPABILITIES_HELP_RESPONSE,
        "messages": [AIMessage(content=CAPABILITIES_HELP_RESPONSE)],
    }


def load_user_profile_node(state: AgentState) -> dict[str, Any]:
    """Load persistent profile memory into graph state."""
    return _profile.load_user_profile_node(state)


def router_node(state: AgentState) -> dict[str, Any]:
    """Classify the latest user query before tool selection."""
    user_query = latest_user_message(state["messages"])
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
    """Run the extracted data-agent loop node."""
    return _loop.data_agent_loop_node(state)


def profile_update_node(state: AgentState) -> dict[str, Any]:
    """Update persistent profile memory after a turn."""
    return _profile.profile_update_node(state)


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
    graph_builder.add_node("capabilities_help_node", capabilities_help_node)
    graph_builder.add_node("router_node", router_node)
    graph_builder.add_node("data_agent_loop_node", data_agent_loop_node)
    graph_builder.add_node("refusal_node", refusal_node)
    graph_builder.add_node("profile_update_node", profile_update_node)

    graph_builder.add_edge(START, "load_user_profile_node")
    graph_builder.add_conditional_edges(
        "load_user_profile_node",
        route_after_profile_load,
        {
            "capabilities_help_node": "capabilities_help_node",
            "router_node": "router_node",
        },
    )
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
    graph_builder.add_edge("capabilities_help_node", END)
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
        logger.warning(
            "Could not read checkpoint state; starting a fresh invocation state. "
            "session_id=%s user_id=%s",
            session_id,
            user_id,
            exc_info=True,
        )
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