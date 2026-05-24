from __future__ import annotations

import json
from typing import Any

from app.agent.context import (
    _build_planner_messages,
    _build_reviewer_messages,
    _compact_tool_input_for_prompt,
)
from app.agent.evidence_contracts import (
    _answer_contract_error,
    _build_final_answer_block_feedback,
    _final_answer_update,
    _return_deterministic_sample_examples_answer_if_ready,
    _return_failed_explicit_resolver_answer_if_ready,
)
from app.agent.followups import _handle_more_examples_follow_up
from app.agent.llm_factory import (
    get_structured_observation_reviewer_llm,
    get_structured_tool_planner_llm,
)
from app.agent.schemas import (
    ObservationReviewDecision,
    ToolPlanDecision,
    VALID_PLANNER_TOOL_NAMES,
)
from app.agent.tool_executor import (
    _execute_selected_tool,
    _tool_call_already_exists,
)
from app.config import settings
from app.state import AgentState


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


def data_agent_loop_node(state: AgentState) -> dict[str, Any]:
    """Run a graph-owned plan -> execute -> review loop for dataset questions."""
    deterministic_result = _handle_more_examples_follow_up(state)
    if deterministic_result is not None:
        return deterministic_result

    reviewer_feedback: str | None = None
    must_call_tool_before_final_answer = False

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
                if must_call_tool_before_final_answer:
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
                    must_call_tool_before_final_answer = True
                    reviewer_feedback = _build_final_answer_block_feedback(
                        contract_error
                    )
                    _debug_trace(
                        f"iteration {iteration_number}/{state['max_iterations']}: "
                        "blocked planner final_answer by evidence contract"
                    )
                    continue

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
                must_call_tool_before_final_answer = False
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
                must_call_tool_before_final_answer = True
                reviewer_feedback = (
                    "The reviewer failed to return a valid structured decision "
                    f"after the latest tool call. Error: {type(exc).__name__}: {exc}. "
                    "Continue agentically from the current tool trace. Do not repeat "
                    "the same failed or already-observed tool call. Because the latest "
                    "observation was not successfully reviewed, do not produce a normal "
                    "answered final answer from it. Choose exactly one next useful tool, "
                    "or produce a cannot-answer style final answer if no useful tool exists."
                )
                _debug_trace(
                    f"iteration {iteration_number}/{state['max_iterations']}: "
                    f"reviewer exception={type(exc).__name__}: {exc}; "
                    "continuing with planner feedback"
                )
                continue

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
                        must_call_tool_before_final_answer = True
                        reviewer_feedback = _build_final_answer_block_feedback(
                            contract_error
                        )
                        _debug_trace(
                            f"iteration {iteration_number}/{state['max_iterations']}: "
                            "blocked reviewer answered by evidence contract"
                        )
                        continue

                final_answer = review.final_answer.strip() or review.reason.strip()
                _debug_trace(
                    f"iteration {iteration_number}/{state['max_iterations']}: "
                    f"returning reviewer status={review.status}"
                )
                return _final_answer_update(state, final_answer)

            if review.status == "needs_more":
                must_call_tool_before_final_answer = True

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
                    must_call_tool_before_final_answer = False
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
                    must_call_tool_before_final_answer = False
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

                _debug_trace(
                    f"iteration {iteration_number}/{state['max_iterations']}: "
                    "reviewer start after direct tool execution"
                )
                try:
                    follow_up_review = _review_observations(state)
                except Exception as exc:
                    must_call_tool_before_final_answer = True
                    reviewer_feedback = (
                        "The reviewer failed to return a valid structured decision "
                        f"after the reviewer-suggested tool. Error: "
                        f"{type(exc).__name__}: {exc}. Because the latest observation "
                        "was not successfully reviewed, do not produce a normal answered "
                        "final answer from it. Choose exactly one next useful tool, or "
                        "produce a cannot-answer style final answer if no useful tool exists."
                    )
                    _debug_trace(
                        f"iteration {iteration_number}/{state['max_iterations']}: "
                        f"follow-up reviewer exception={type(exc).__name__}: {exc}; "
                        "continuing with planner feedback"
                    )
                    continue

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
                            must_call_tool_before_final_answer = True
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
                    must_call_tool_before_final_answer = True
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