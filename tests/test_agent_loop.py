from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from app.agent import loop
from app.agent import schemas


def _state(query: str = "How many refund requests?") -> dict[str, Any]:
    return {
        "messages": [HumanMessage(content=query)],
        "session_id": "test_session",
        "user_id": "max",
        "route": "structured",
        "route_reason": "The user asks for a dataset operation.",
        "tool_trace": [],
        "last_structured_results": [],
        "user_profile": "# User Profile\n",
        "max_iterations": 12,
        "final_answer": None,
    }


def _tool_events(tool_trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return only actual tool-call events from a mixed reasoning trace."""
    return [
        item for item in tool_trace
        if item.get("event_type", "tool") == "tool"
    ]


def _reviewer_events(tool_trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return only reviewer-decision events from a mixed reasoning trace."""
    return [
        item for item in tool_trace
        if item.get("event_type") == "reviewer"
    ]


class FakePlannerLLM:
    def __init__(self, *decisions: schemas.ToolPlanDecision) -> None:
        self.decisions = list(decisions)
        self.received_messages = []

    def invoke(self, messages):
        self.received_messages.append(messages)
        if not self.decisions:
            raise AssertionError("Planner received more calls than expected.")
        return self.decisions.pop(0)


class FakeReviewerLLM:
    def __init__(self, *decisions: schemas.ObservationReviewDecision) -> None:
        self.decisions = list(decisions)
        self.received_messages = []

    def invoke(self, messages):
        self.received_messages.append(messages)
        if not self.decisions:
            raise AssertionError("Reviewer received more calls than expected.")
        return self.decisions.pop(0)


class FailingPlannerLLM:
    def invoke(self, messages):
        raise RuntimeError("Simulated planner failure.")


class FailingReviewerLLM:
    def invoke(self, messages):
        raise RuntimeError("Simulated reviewer failure.")


def _plan_call(tool_name: str, tool_input: dict[str, Any]) -> schemas.ToolPlanDecision:
    return schemas.ToolPlanDecision(
        action="call_tool",
        tool_name=tool_name,
        tool_input=tool_input,
        reason=f"Call {tool_name}.",
    )


def _plan_final(answer: str) -> schemas.ToolPlanDecision:
    return schemas.ToolPlanDecision(
        action="final_answer",
        final_answer=answer,
        reason="The current observations are sufficient.",
    )


def _review_answer(answer: str) -> schemas.ObservationReviewDecision:
    return schemas.ObservationReviewDecision(
        status="answered",
        reason="The observations are sufficient.",
        final_answer=answer,
    )


def _review_needs_more(
    reason: str,
    suggested_tool_name: str,
    suggested_tool_input: dict[str, Any],
) -> schemas.ObservationReviewDecision:
    return schemas.ObservationReviewDecision(
        status="needs_more",
        reason=reason,
        suggested_tool_name=suggested_tool_name,
        suggested_tool_input=suggested_tool_input,
    )


def test_loop_returns_planner_final_answer_without_tools(monkeypatch) -> None:
    planner = FakePlannerLLM(
        _plan_final("The total count of the last two results is 1,356.")
    )

    monkeypatch.setattr(loop, "get_structured_tool_planner_llm", lambda: planner)

    result = loop.data_agent_loop_node(
        _state("What is the total count of the last two?")
    )

    assert result["final_answer"] == "The total count of the last two results is 1,356."
    assert isinstance(result["messages"][0], AIMessage)
    assert result["tool_trace"] == []


def test_loop_executes_tool_then_returns_reviewer_answer(monkeypatch) -> None:
    state = _state("How many refund requests?")

    planner = FakePlannerLLM(
        _plan_call(
            "count_rows",
            {
                "category": "REFUND",
            },
        )
    )
    reviewer = FakeReviewerLLM(
        _review_answer("There are 2,992 refund-request rows in the dataset.")
    )

    def fake_execute_selected_tool(
        state,
        tool_name: str,
        tool_input: dict[str, Any],
    ) -> None:
        state["tool_trace"].append(
            {
                "event_type": "tool",
                "tool_name": tool_name,
                "tool_input": {
                    "category": "REFUND",
                    "intent": None,
                    "text_query": None,
                },
                "observation": (
                    '{"count": 2992, "applied_filters": '
                    '{"category": "REFUND", "intent": null, "text_query": null}}'
                ),
            }
        )
        state["last_structured_results"].append(
            {
                "label": "count_rows",
                "value": 2992,
                "query_type": "count",
                "filters": {
                    "category": "REFUND",
                    "intent": None,
                    "text_query": None,
                },
                "match_count": 2992,
            }
        )

    monkeypatch.setattr(loop, "get_structured_tool_planner_llm", lambda: planner)
    monkeypatch.setattr(
        loop,
        "get_structured_observation_reviewer_llm",
        lambda: reviewer,
    )
    monkeypatch.setattr(loop, "_execute_selected_tool", fake_execute_selected_tool)

    result = loop.data_agent_loop_node(state)

    assert result["final_answer"] == (
        "There are 2,992 refund-request rows in the dataset."
    )
    assert _tool_events(state["tool_trace"]) == [
        {
            "event_type": "tool",
            "tool_name": "count_rows",
            "tool_input": {
                "category": "REFUND",
                "intent": None,
                "text_query": None,
            },
            "observation": (
                '{"count": 2992, "applied_filters": '
                '{"category": "REFUND", "intent": null, "text_query": null}}'
            ),
        }
    ]
    assert _reviewer_events(state["tool_trace"]) == [
        {
            "event_type": "reviewer",
            "reviewer_status": "answered",
            "reviewer_reason": "The observations are sufficient.",
            "reviewer_final_answer": (
                "There are 2,992 refund-request rows in the dataset."
            ),
            "suggested_tool_name": "",
            "suggested_tool_input": {},
        }
    ]


def test_loop_returns_reviewer_suggestion_to_planner(monkeypatch) -> None:
    state = _state("How many refund requests?")

    planner = FakePlannerLLM(
        _plan_call(
            "resolve_filter_value",
            {
                "query": "refund requests",
                "columns": ["category", "intent"],
                "top_k": 5,
            },
        ),
        _plan_call(
            "count_rows",
            {
                "category": "REFUND",
            },
        ),
    )
    reviewer = FakeReviewerLLM(
        _review_needs_more(
            reason="Need count_rows after resolving the filter.",
            suggested_tool_name="count_rows",
            suggested_tool_input={
                "category": "REFUND",
            },
        ),
        _review_answer("There are 2,992 refund-request rows in the dataset."),
    )

    executed_tools: list[tuple[str, dict[str, Any]]] = []

    def fake_execute_selected_tool(
        state,
        tool_name: str,
        tool_input: dict[str, Any],
    ) -> None:
        executed_tools.append((tool_name, tool_input))
        state["tool_trace"].append(
            {
                "event_type": "tool",
                "tool_name": tool_name,
                "tool_input": tool_input,
                "observation": "{}",
            }
        )

    monkeypatch.setattr(loop, "get_structured_tool_planner_llm", lambda: planner)
    monkeypatch.setattr(
        loop,
        "get_structured_observation_reviewer_llm",
        lambda: reviewer,
    )
    monkeypatch.setattr(loop, "_execute_selected_tool", fake_execute_selected_tool)

    result = loop.data_agent_loop_node(state)

    assert executed_tools == [
        (
            "resolve_filter_value",
            {
                "query": "refund requests",
                "columns": ["category", "intent"],
                "top_k": 5,
            },
        ),
        (
            "count_rows",
            {
                "category": "REFUND",
            },
        ),
    ]
    assert result["final_answer"] == (
        "There are 2,992 refund-request rows in the dataset."
    )
    reviewer_events = _reviewer_events(state["tool_trace"])
    assert [event["reviewer_status"] for event in reviewer_events] == [
        "needs_more",
        "answered",
    ]
    assert len(planner.received_messages) == 2
    assert reviewer_events[0]["suggested_tool_name"] == "count_rows"
    assert reviewer_events[0]["suggested_tool_input"] == {
        "category": "REFUND",
    }
    assert reviewer_events[1]["reviewer_final_answer"] == (
        "There are 2,992 refund-request rows in the dataset."
    )
    assert "The reviewer only suggests the next step" in (
        planner.received_messages[1][1].content
    )


def test_loop_blocks_duplicate_planner_tool_call_and_uses_existing_observation(
    monkeypatch,
) -> None:
    state = _state("How many refund requests?")
    state["tool_trace"] = [
        {
            "event_type": "tool",
            "tool_name": "count_rows",
            "tool_input": {
                "category": "REFUND",
                "intent": None,
                "text_query": None,
            },
            "observation": '{"count": 2992}',
        }
    ]

    planner = FakePlannerLLM(
        _plan_call(
            "count_rows",
            {
                "category": "REFUND",
            },
        ),
        _plan_final("There are 2,992 refund-request rows in the dataset."),
    )

    def fail_if_tool_executed(state, tool_name: str, tool_input: dict[str, Any]) -> None:
        raise AssertionError("Duplicate tool call should be blocked.")

    monkeypatch.setattr(loop, "get_structured_tool_planner_llm", lambda: planner)
    monkeypatch.setattr(loop, "_execute_selected_tool", fail_if_tool_executed)

    result = loop.data_agent_loop_node(state)

    assert result["final_answer"] == (
        "There are 2,992 refund-request rows in the dataset."
    )


def test_loop_blocks_planner_final_answer_when_contract_requires_tool(
    monkeypatch,
) -> None:
    state = _state("What is the distribution of intents in the ACCOUNT category?")

    planner = FakePlannerLLM(
        _plan_final("Premature distribution answer."),
        _plan_call(
            "group_counts",
            {
                "group_by": "intent",
                "category": "ACCOUNT",
                "top_k": 20,
            },
        ),
    )
    reviewer = FakeReviewerLLM(
        _review_answer("The ACCOUNT category has recover_password and delete_account.")
    )

    executed_tools: list[tuple[str, dict[str, Any]]] = []

    def fake_answer_contract_error(state) -> str | None:
        if not state["tool_trace"]:
            return "Missing scoped group_counts evidence."
        return None

    def fake_execute_selected_tool(
        state,
        tool_name: str,
        tool_input: dict[str, Any],
    ) -> None:
        executed_tools.append((tool_name, tool_input))
        state["tool_trace"].append(
            {
                "event_type": "tool",
                "tool_name": tool_name,
                "tool_input": tool_input,
                "observation": "{}",
            }
        )

    monkeypatch.setattr(loop, "get_structured_tool_planner_llm", lambda: planner)
    monkeypatch.setattr(
        loop,
        "get_structured_observation_reviewer_llm",
        lambda: reviewer,
    )
    monkeypatch.setattr(loop, "_answer_contract_error", fake_answer_contract_error)
    monkeypatch.setattr(loop, "_execute_selected_tool", fake_execute_selected_tool)

    result = loop.data_agent_loop_node(state)

    assert executed_tools == [
        (
            "group_counts",
            {
                "group_by": "intent",
                "category": "ACCOUNT",
                "top_k": 20,
            },
        )
    ]
    assert result["final_answer"] == (
        "The ACCOUNT category has recover_password and delete_account."
    )


def test_loop_returns_fallback_on_planner_exception(monkeypatch) -> None:
    monkeypatch.setattr(
        loop,
        "get_structured_tool_planner_llm",
        lambda: FailingPlannerLLM(),
    )

    result = loop.data_agent_loop_node(_state("Analyze the dataset."))

    assert result["final_answer"] == (
        "I could not complete the analysis within the allowed number of "
        "reasoning steps. Please try asking a more specific dataset question."
    )


def test_loop_continues_after_reviewer_exception(monkeypatch) -> None:
    state = _state("How many refund requests?")

    planner = FakePlannerLLM(
        _plan_call(
            "count_rows",
            {
                "category": "REFUND",
            },
        ),
        _plan_final("There are 2,992 refund-request rows in the dataset."),
    )

    def fake_execute_selected_tool(
        state,
        tool_name: str,
        tool_input: dict[str, Any],
    ) -> None:
        state["tool_trace"].append(
            {
                "event_type": "tool",
                "tool_name": tool_name,
                "tool_input": tool_input,
                "observation": '{"count": 2992}',
            }
        )

    monkeypatch.setattr(loop, "get_structured_tool_planner_llm", lambda: planner)
    monkeypatch.setattr(
        loop,
        "get_structured_observation_reviewer_llm",
        lambda: FailingReviewerLLM(),
    )
    monkeypatch.setattr(loop, "_execute_selected_tool", fake_execute_selected_tool)

    result = loop.data_agent_loop_node(state)

    assert result["final_answer"] == (
        "There are 2,992 refund-request rows in the dataset."
    )
    reviewer_events = _reviewer_events(state["tool_trace"])
    assert reviewer_events == [
        {
            "event_type": "reviewer",
            "reviewer_status": "error",
            "reviewer_reason": (
                "Reviewer failed to return a valid structured decision after "
                "the latest tool call: RuntimeError: Simulated reviewer failure."
            ),
            "reviewer_final_answer": "",
            "suggested_tool_name": "",
            "suggested_tool_input": {},
        }
    ]