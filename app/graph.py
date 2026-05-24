from __future__ import annotations

import sqlite3
from functools import lru_cache
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph

from app.config import settings
from app.agent import loop as _loop
from app.agent import profile as _profile
from app.agent import evidence_contracts as _evidence_contracts
from app.agent import followups as _followups
from app.agent import tool_executor as _tool_executor
from app.memory import read_user_profile_impl, update_user_profile_impl
from app.agent.llm_factory import (
    _create_agent_chat_llm,
    get_agent_llm,
    get_structured_observation_reviewer_llm,
    get_structured_profile_llm,
    get_structured_tool_planner_llm,
)
from app.agent.context import (
    _build_planner_messages,
    _build_reviewer_messages,
    _compact_tool_input_for_prompt,
    _compact_tool_trace_for_prompt,
    _latest_user_message,
    _profile_context_for_planner,
    _structured_results_for_prompt,
)
from app.prompts import OUT_OF_SCOPE_REFUSAL
from app.agent.schemas import (
    ObservationReviewDecision,
    PlannerToolName,
    ProfileObservationDecision,
    ToolPlanDecision,
    VALID_PLANNER_TOOL_NAMES,
)

from app.router import RouteDecision, route_query_with_reason
from app.state import AgentState, AnalysisResult, ToolTraceItem
from app.tools import (
    count_rows_impl,
    get_dataset_schema_impl,
    group_counts_impl,
    resolve_filter_value_impl,
    sample_examples_impl,
    summarize_rows_impl,
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


_append_trace = _tool_executor._append_trace
_append_tool_error = _tool_executor._append_tool_error
_append_structured_result = _tool_executor._append_structured_result
_normalize_tool_input = _tool_executor._normalize_tool_input
_normalize_resolve_filter_value_input = (
    _tool_executor._normalize_resolve_filter_value_input
)
_tool_filters = _tool_executor._tool_filters
_canonical_tool_input = _tool_executor._canonical_tool_input
_tool_call_already_exists = _tool_executor._tool_call_already_exists
_format_model_dict = _tool_executor._format_model_dict
_requires_grouped_filtered_scope = _tool_executor._requires_grouped_filtered_scope

_final_answer_update = _evidence_contracts._final_answer_update
_is_example_request = _evidence_contracts._is_example_request
_trace_observation_is_error = _evidence_contracts._trace_observation_is_error
_failed_explicit_resolver_final_answer = (
    _evidence_contracts._failed_explicit_resolver_final_answer
)
_return_failed_explicit_resolver_answer_if_ready = (
    _evidence_contracts._return_failed_explicit_resolver_answer_if_ready
)
_sample_examples_observation_has_examples = (
    _evidence_contracts._sample_examples_observation_has_examples
)
_return_deterministic_sample_examples_answer_if_ready = (
    _evidence_contracts._return_deterministic_sample_examples_answer_if_ready
)
_has_filtered_group_counts_trace = _evidence_contracts._has_filtered_group_counts_trace
_has_resolver_trace_for_column = _evidence_contracts._has_resolver_trace_for_column
_parse_trace_observation_json = _evidence_contracts._parse_trace_observation_json
_normalize_contract_text = _evidence_contracts._normalize_contract_text
_is_text_query_only_summary = _evidence_contracts._is_text_query_only_summary
_summarize_rows_uses_resolved_filter = (
    _evidence_contracts._summarize_rows_uses_resolved_filter
)
_semantic_summary_contract_error = _evidence_contracts._semantic_summary_contract_error
_answer_contract_error = _evidence_contracts._answer_contract_error
_build_final_answer_block_feedback = (
    _evidence_contracts._build_final_answer_block_feedback
)
_is_more_examples_query = _followups._is_more_examples_query
_requested_example_count = _followups._requested_example_count
_latest_sample_context = _followups._latest_sample_context


def _sync_tool_executor_dependencies() -> None:
    """Keep temporary graph-level monkeypatch compatibility during extraction.

    Current graph tests monkeypatch tool implementations on app.graph and then
    call graph._execute_selected_tool. Until tests are migrated to
    app.agent.tool_executor, copy graph-level patched callables into the
    extracted executor module immediately before execution.
    """
    _tool_executor.count_rows_impl = count_rows_impl
    _tool_executor.get_dataset_schema_impl = get_dataset_schema_impl
    _tool_executor.group_counts_impl = group_counts_impl
    _tool_executor.resolve_filter_value_impl = resolve_filter_value_impl
    _tool_executor.sample_examples_impl = sample_examples_impl
    _tool_executor.summarize_rows_impl = summarize_rows_impl
    _tool_executor.read_user_profile_impl = read_user_profile_impl


def _format_sample_examples_observation(
    filters: dict[str, str | None],
    n: int,
    offset: int,
) -> tuple[str, int, int]:
    """Compatibility wrapper for the extracted sample-example formatter."""
    _sync_tool_executor_dependencies()
    return _tool_executor._format_sample_examples_observation(
        filters=filters,
        n=n,
        offset=offset,
    )


def _execute_selected_tool(
        state: AgentState,
        tool_name: str,
        tool_input: dict[str, Any],
) -> None:
    """Compatibility wrapper for the extracted tool executor."""
    _sync_tool_executor_dependencies()
    _tool_executor._execute_selected_tool(
        state=state,
        tool_name=tool_name,
        tool_input=tool_input,
    )


def _handle_more_examples_follow_up(state: AgentState) -> dict[str, Any] | None:
    """Compatibility wrapper for deterministic example follow-ups."""
    return _followups._handle_more_examples_follow_up(
        state,
        sample_formatter=lambda filters, n, offset: _format_sample_examples_observation(
            filters=filters,
            n=n,
            offset=offset,
        ),
    )


def _review_observations(
    state: AgentState,
) -> ObservationReviewDecision:
    """Ask the reviewer whether the current observations answer the query."""
    reviewer_llm = get_structured_observation_reviewer_llm()
    review = reviewer_llm.invoke(_build_reviewer_messages(state))
    if not isinstance(review, ObservationReviewDecision):
        review = ObservationReviewDecision.model_validate(review)
    return review


_fallback_answer = _loop._fallback_answer
_debug_trace_enabled = _loop._debug_trace_enabled
_debug_trace = _loop._debug_trace


def _sync_loop_dependencies() -> None:
    """Keep graph-level monkeypatch compatibility during loop extraction.

    Current tests patch planner/reviewer/tool functions on app.graph. The
    extracted loop module imports its own dependencies, so this bridge copies
    graph-level patched callables into app.agent.loop immediately before the
    loop runs.
    """
    _sync_tool_executor_dependencies()

    _loop.get_structured_tool_planner_llm = get_structured_tool_planner_llm
    _loop.get_structured_observation_reviewer_llm = (
        get_structured_observation_reviewer_llm
    )

    _loop._build_planner_messages = _build_planner_messages
    _loop._build_reviewer_messages = _build_reviewer_messages
    _loop._compact_tool_input_for_prompt = _compact_tool_input_for_prompt

    _loop._execute_selected_tool = _execute_selected_tool
    _loop._tool_call_already_exists = _tool_call_already_exists
    _loop._handle_more_examples_follow_up = _handle_more_examples_follow_up
    _loop._review_observations = _review_observations

    _loop._answer_contract_error = _answer_contract_error
    _loop._build_final_answer_block_feedback = _build_final_answer_block_feedback
    _loop._final_answer_update = _final_answer_update
    _loop._return_failed_explicit_resolver_answer_if_ready = (
        _return_failed_explicit_resolver_answer_if_ready
    )
    _loop._return_deterministic_sample_examples_answer_if_ready = (
        _return_deterministic_sample_examples_answer_if_ready
    )


def _sync_profile_dependencies() -> None:
    """Keep graph-level monkeypatch compatibility during profile extraction."""
    _profile.read_user_profile_impl = read_user_profile_impl
    _profile.update_user_profile_impl = update_user_profile_impl
    _profile.get_structured_profile_llm = get_structured_profile_llm
    _profile.ProfileObservationDecision = ProfileObservationDecision
    _profile._latest_user_message = _latest_user_message


def load_user_profile_node(state: AgentState) -> dict[str, Any]:
    """Compatibility wrapper for the extracted profile-load node."""
    _sync_profile_dependencies()
    return _profile.load_user_profile_node(state)


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
    """Compatibility wrapper for the extracted data-agent loop."""
    _sync_loop_dependencies()
    return _loop.data_agent_loop_node(state)


def profile_update_node(state: AgentState) -> dict[str, Any]:
    """Compatibility wrapper for the extracted profile-update node."""
    _sync_profile_dependencies()
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