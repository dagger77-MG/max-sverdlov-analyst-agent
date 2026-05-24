from __future__ import annotations

import json
import sqlite3
from functools import lru_cache
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from app.config import settings
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
from app.prompts import (
    OUT_OF_SCOPE_REFUSAL,
    PROFILE_UPDATE_SYSTEM_PROMPT,
)
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


def _fallback_answer() -> str:
    """Return a safe graph fallback for planner/reviewer/tool errors."""
    return (
        "I could not complete the analysis within the allowed number of "
        "reasoning steps. Please try asking a more specific dataset question."
    )


def _debug_trace_enabled() -> bool:
    """Return True when local live graph-loop debug output is enabled."""
    value = settings.debug_trace
    return value


def _debug_trace(message: str) -> None:
    """Print live graph-loop debug events for local development.

    This is intentionally separate from the user-facing tool_trace. It helps
    debug planner/reviewer loops and swallowed exceptions while the agent is
    still running.
    """
    if _debug_trace_enabled():
        print(f"[debug] {message}", flush=True)


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
    reviewer_requires_tool = False

    for iteration_index in range(state["max_iterations"]):
        iteration_number = iteration_index + 1
        try:
            _debug_trace(
                f"iteration {iteration_number}/{state['max_iterations']}: "
                "planner start"
            )
            planner_llm = get_structured_tool_planner_llm()
            plan = planner_llm.invoke(
                _build_planner_messages(state, reviewer_feedback)
            )
            if not isinstance(plan, ToolPlanDecision):
                plan = ToolPlanDecision.model_validate(plan)

            _debug_trace(
                f"iteration {iteration_number}/{state['max_iterations']}: "
                f"planner action={plan.action}; "
                f"tool={plan.tool_name or '-'}; "
                f"reason={plan.reason}"
            )

            if plan.action == "final_answer":
                if reviewer_requires_tool:
                    reviewer_feedback = (
                        "The previous reviewer decision was needs_more, so the "
                        "current observations are not sufficient for a final answer. "
                        "Call exactly one valid next tool. If the user-provided "
                        "category or intent value was not resolved yet, call "
                        "resolve_filter_value with the columns implied by the user's wording."
                    )
                    _debug_trace(
                        f"iteration {iteration_number}/{state['max_iterations']}: "
                        "blocked planner final_answer after reviewer needs_more"
                    )
                    continue

                final_answer = plan.final_answer.strip()
                if not final_answer:
                    reviewer_feedback = (
                        "Planner chose final_answer but returned empty text."
                    )
                    _debug_trace(
                        f"iteration {iteration_number}/{state['max_iterations']}: "
                        "planner returned empty final_answer"
                    )
                    continue
                contract_error = _answer_contract_error(state)
                if contract_error:
                    reviewer_requires_tool = True
                    reviewer_feedback = _build_final_answer_block_feedback(
                        contract_error
                    )
                    _debug_trace(
                        f"iteration {iteration_number}/{state['max_iterations']}: "
                        "blocked planner final_answer by evidence contract"
                    )
                    continue

                state["final_answer"] = final_answer
                _debug_trace(
                    f"iteration {iteration_number}/{state['max_iterations']}: "
                    "returning planner final_answer"
                )
                return _final_answer_update(state, final_answer)

            if (
                plan.action == "call_tool"
                and _tool_call_already_exists(
                    state=state,
                    tool_name=plan.tool_name,
                    tool_input=plan.tool_input,
                )
            ):
                reviewer_requires_tool = False
                reviewer_feedback = (
                    "This exact tool call already exists in the current turn trace. "
                    "Do not repeat it. Use the existing observation to produce a "
                    "final answer or a cannot-answer style final answer."
                )
                _debug_trace(
                    f"iteration {iteration_number}/{state['max_iterations']}: "
                    f"blocked duplicate planner tool call={plan.tool_name}"
                )
                continue

            _execute_selected_tool(
                state=state,
                tool_name=plan.tool_name,
                tool_input=plan.tool_input,
            )
            _debug_trace(
                f"iteration {iteration_number}/{state['max_iterations']}: "
                f"executed tool={plan.tool_name}; "
                f"trace_steps={len(state['tool_trace'])}"
            )

            failed_resolver_answer = (
                _return_failed_explicit_resolver_answer_if_ready(state)
            )
            if failed_resolver_answer is not None:
                _debug_trace(
                    f"iteration {iteration_number}/{state['max_iterations']}: "
                    "returning deterministic failed resolver answer"
                )
                return failed_resolver_answer


            deterministic_sample_answer = (
                _return_deterministic_sample_examples_answer_if_ready(
                    state=state,
                    tool_name=plan.tool_name,
                )
            )
            if deterministic_sample_answer is not None:
                _debug_trace(
                    f"iteration {iteration_number}/{state['max_iterations']}: "
                    "returning deterministic sample_examples answer"
                )
                return deterministic_sample_answer

            _debug_trace(
                f"iteration {iteration_number}/{state['max_iterations']}: "
                "reviewer start"
            )
            try:
                review = _review_observations(state)
            except Exception as exc:
                reviewer_requires_tool = False
                reviewer_feedback = (
                    "The reviewer failed to return a valid structured decision "
                    f"after the latest tool call. Error: {type(exc).__name__}: {exc}. "
                    "Continue agentically from the current tool trace. Do not repeat "
                    "the same failed or already-observed tool call. If the current "
                    "observations fully answer the exact user request, produce a "
                    "grounded final answer. Otherwise, choose exactly one next useful "
                    "tool based on the current observations."
                )
                _debug_trace(
                    f"iteration {iteration_number}/{state['max_iterations']}: "
                    f"reviewer exception={type(exc).__name__}: {exc}; "
                    "continuing with planner feedback"
                )
                continue

            reviewer_feedback = review.reason
            _debug_trace(
                f"iteration {iteration_number}/{state['max_iterations']}: "
                f"reviewer status={review.status}; "
                f"suggested_tool={review.suggested_tool_name or '-'}; "
                f"reason={review.reason}"
            )

            if review.status in {"answered", "cannot_answer"}:
                if review.status == "answered":
                    contract_error = _answer_contract_error(state)
                    if contract_error:
                        reviewer_requires_tool = True
                        reviewer_feedback = _build_final_answer_block_feedback(
                            contract_error
                        )
                        _debug_trace(
                            f"iteration {iteration_number}/{state['max_iterations']}: "
                            "blocked reviewer answered by evidence contract"
                        )
                        continue

                reviewer_requires_tool = False
                final_answer = review.final_answer.strip() or review.reason.strip()
                _debug_trace(
                    f"iteration {iteration_number}/{state['max_iterations']}: "
                    f"returning reviewer status={review.status}"
                )
                return _final_answer_update(state, final_answer)

            if review.status == "needs_more":
                reviewer_requires_tool = True

                if not review.suggested_tool_name:
                    reviewer_feedback = (
                        f"{review.reason}\n"
                        "Reviewer returned needs_more but did not provide a suggested tool. "
                        "Choose exactly one valid next tool yourself. For unresolved "
                        "category/intent filters, use resolve_filter_value first."
                    )
                    _debug_trace(
                        f"iteration {iteration_number}/{state['max_iterations']}: "
                        "reviewer returned needs_more without suggested tool"
                    )
                    continue

                if review.suggested_tool_name not in VALID_PLANNER_TOOL_NAMES:
                    reviewer_requires_tool = False
                    reviewer_feedback = (
                        f"{review.reason}\n"
                        f"Reviewer returned needs_more with invalid suggested_tool_name="
                        f"{review.suggested_tool_name!r}. This is not a callable tool. "
                        "Do not call another tool only because of this malformed reviewer "
                        "decision. If the existing observations are enough, produce a "
                        "final answer. If the requested subset/value does not exist, "
                        "produce a cannot-answer style final answer."
                    )
                    _debug_trace(
                        f"iteration {iteration_number}/{state['max_iterations']}: "
                        f"reviewer suggested invalid tool={review.suggested_tool_name!r}"
                    )
                    continue

                if _tool_call_already_exists(
                    state=state,
                    tool_name=review.suggested_tool_name,
                    tool_input=review.suggested_tool_input,
                ):
                    reviewer_requires_tool = False
                    reviewer_feedback = (
                        f"{review.reason}\n"
                        "The reviewer suggested a tool call that already exists in "
                        "the current turn trace. Do not repeat the same tool call. "
                        "Use the existing observation to produce a final answer or "
                        "a cannot-answer style final answer."
                    )
                    _debug_trace(
                        f"iteration {iteration_number}/{state['max_iterations']}: "
                        "reviewer suggested duplicate tool call"
                    )
                    continue

                _debug_trace(
                    f"iteration {iteration_number}/{state['max_iterations']}: "
                    f"reviewer requested next tool={review.suggested_tool_name}"
                    "\nexecuting directly"
                )

                _execute_selected_tool(
                    state=state,
                    tool_name=review.suggested_tool_name,
                    tool_input=review.suggested_tool_input,
                )

                _debug_trace(
                    f"iteration {iteration_number}/{state['max_iterations']}: "
                    f"executed reviewer tool={review.suggested_tool_name}; "
                    f"trace_steps={len(state['tool_trace'])}"
                )

                failed_resolver_answer = (
                    _return_failed_explicit_resolver_answer_if_ready(state)
                )
                if failed_resolver_answer is not None:
                    _debug_trace(
                        f"iteration {iteration_number}/{state['max_iterations']}: "
                        "returning deterministic reviewer failed resolver answer"
                    )
                    return failed_resolver_answer

                deterministic_sample_answer = (
                    _return_deterministic_sample_examples_answer_if_ready(
                        state=state,
                        tool_name=review.suggested_tool_name,
                    )
                )
                if deterministic_sample_answer is not None:
                    _debug_trace(
                        f"iteration {iteration_number}/{state['max_iterations']}: "
                        "returning deterministic reviewer sample_examples answer"
                    )
                    return deterministic_sample_answer

                reviewer_requires_tool = False
                reviewer_feedback = (
                    f"The reviewer-suggested tool {review.suggested_tool_name} "
                    "has now been executed. The observation reviewer will inspect "
                    "the latest observation before another planner step is allowed."
                )

                _debug_trace(
                    f"iteration {iteration_number}/{state['max_iterations']}: "
                    "reviewer start after direct tool execution"
                )
                try:
                    follow_up_review = _review_observations(state)
                except Exception as exc:
                    reviewer_feedback = (
                        "The reviewer failed to return a valid structured decision "
                        f"after the reviewer-suggested tool. Error: "
                        f"{type(exc).__name__}: {exc}. Continue agentically from "
                        "the current tool trace. Do not repeat the same tool call."
                    )
                    _debug_trace(
                        f"iteration {iteration_number}/{state['max_iterations']}: "
                        f"follow-up reviewer exception={type(exc).__name__}: {exc}; "
                        "continuing with planner feedback"
                    )
                    continue

                reviewer_feedback = follow_up_review.reason
                _debug_trace(
                    f"iteration {iteration_number}/{state['max_iterations']}: "
                    f"follow-up reviewer status={follow_up_review.status}; "
                    f"suggested_tool={follow_up_review.suggested_tool_name or '-'}; "
                    f"reason={follow_up_review.reason}"
                )

                if follow_up_review.status in {"answered", "cannot_answer"}:
                    if follow_up_review.status == "answered":
                        contract_error = _answer_contract_error(state)
                        if contract_error:
                            reviewer_requires_tool = True
                            reviewer_feedback = _build_final_answer_block_feedback(
                                contract_error
                            )
                            _debug_trace(
                                f"iteration {iteration_number}/{state['max_iterations']}: "
                                "blocked follow-up reviewer answered by evidence contract"
                            )
                            continue

                    final_answer = (
                        follow_up_review.final_answer.strip()
                        or follow_up_review.reason.strip()
                    )
                    _debug_trace(
                        f"iteration {iteration_number}/{state['max_iterations']}: "
                        f"returning follow-up reviewer status={follow_up_review.status}"
                    )
                    return _final_answer_update(state, final_answer)

                if follow_up_review.status == "needs_more":
                    reviewer_requires_tool = True
                    reviewer_feedback = (
                        f"{follow_up_review.reason}\n"
                        f"Suggested next tool: {follow_up_review.suggested_tool_name}\n"
                        f"Suggested next input: "
                        f"{json.dumps(_compact_tool_input_for_prompt(follow_up_review.suggested_tool_input), ensure_ascii=False, default=str)}"
                    )
                    continue

        except Exception as exc:
            fallback = _fallback_answer()
            _debug_trace(
                f"iteration {iteration_number}/{state['max_iterations']}: "
                f"exception={type(exc).__name__}: {exc}"
            )
            return _final_answer_update(state, fallback)

    fallback = _fallback_answer()
    _debug_trace(
        f"max_iterations_exhausted={state['max_iterations']}; "
        f"trace_steps={len(state['tool_trace'])}"
    )

    return _final_answer_update(state, fallback)


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