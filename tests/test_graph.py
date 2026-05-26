from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app import graph
from app.agent import context as agent_context
from app.agent import followups as agent_followups
from app.agent import loop as agent_loop
from app.agent import profile as agent_profile
from app.agent import schemas
from app.agent import tool_executor
from app.logging_utils import format_reasoning_trace


@pytest.fixture(autouse=True)
def patch_checkpointer(monkeypatch: pytest.MonkeyPatch):
    """Keep graph unit tests independent from optional SQLite checkpoints."""
    graph.build_graph.cache_clear()

    if hasattr(graph.get_checkpointer, "cache_clear"):
        graph.get_checkpointer.cache_clear()

    if hasattr(agent_loop.get_structured_tool_planner_llm, "cache_clear"):
        agent_loop.get_structured_tool_planner_llm.cache_clear()

    if hasattr(agent_loop.get_structured_observation_reviewer_llm, "cache_clear"):
        agent_loop.get_structured_observation_reviewer_llm.cache_clear()

    if hasattr(agent_profile.get_structured_profile_llm, "cache_clear"):
        agent_profile.get_structured_profile_llm.cache_clear()

    monkeypatch.setattr(graph, "get_checkpointer", lambda: None)

    yield

    graph.build_graph.cache_clear()


class FakeProfileLLM:
    def __init__(self, observation: str = "") -> None:
        self.observation = observation

    def invoke(self, messages):
        return schemas.ProfileObservationDecision(observation=self.observation)


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


def _profile_result(user_id: str, profile: str = "# User Profile\n"):
    return type(
        "ProfileResult",
        (),
        {
            "user_id": user_id,
            "profile": profile,
        },
    )()


def _model_result(**values):
    return type(
        "ModelResult",
        (),
        {
            **values,
            "model_dump": lambda self: values,
        },
    )()


def _example_row(
    row_id: int,
    instruction: str,
    response: str,
    category: str,
    intent: str,
):
    return type(
        "ExampleRow",
        (),
        {
            "row_id": row_id,
            "instruction": instruction,
            "response": response,
            "category": category,
            "intent": intent,
        },
    )()


def _plan_call(tool_name: str, tool_input: dict) -> schemas.ToolPlanDecision:
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
        reason="The available context is sufficient.",
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
    suggested_tool_input: dict,
) -> schemas.ObservationReviewDecision:
    return schemas.ObservationReviewDecision(
        status="needs_more",
        reason=reason,
        suggested_tool_name=suggested_tool_name,
        suggested_tool_input=suggested_tool_input,
    )


def _tool_events(tool_trace: list[dict]) -> list[dict]:
    """Return only actual tool-call events from a mixed reasoning trace."""
    return [
        item for item in tool_trace
        if item.get("event_type", "tool") == "tool"
    ]


def _reviewer_events(tool_trace: list[dict]) -> list[dict]:
    """Return only reviewer-decision events from a mixed reasoning trace."""
    return [
        item for item in tool_trace
        if item.get("event_type") == "reviewer"
    ]


def test_observation_review_decision_clears_suggested_tool_when_answered() -> None:
    decision = schemas.ObservationReviewDecision(
        status="answered",
        reason="The observations are sufficient.",
        final_answer="There are 2,992 refund rows.",
        suggested_tool_name="count_rows",
        suggested_tool_input={"category": "REFUND"},
    )

    assert decision.suggested_tool_name == ""
    assert decision.suggested_tool_input == {}
    assert decision.final_answer == "There are 2,992 refund rows."


def test_observation_review_decision_clears_final_answer_when_needs_more() -> None:
    decision = schemas.ObservationReviewDecision(
        status="needs_more",
        reason="A count_rows call is still needed.",
        final_answer="Premature answer.",
        suggested_tool_name="count_rows",
        suggested_tool_input={"category": "REFUND"},
    )

    assert decision.final_answer == ""
    assert decision.suggested_tool_name == "count_rows"
    assert decision.suggested_tool_input == {"category": "REFUND"}


def test_format_reasoning_trace_renders_reviewer_decisions() -> None:
    trace = format_reasoning_trace(
        route="structured",
        route_reason="The user asks for examples from a dataset category.",
        tool_trace=[
            {
                "event_type": "tool",
                "tool_name": "resolve_filter_value",
                "tool_input": {
                    "query": "SHIPPING",
                    "columns": ["category"],
                    "top_k": 5,
                },
                "observation": '{"confidence":"high"}',
            },
            {
                "event_type": "reviewer",
                "reviewer_status": "needs_more",
                "reviewer_reason": "Examples require sample_examples output.",
                "reviewer_final_answer": "",
                "suggested_tool_name": "sample_examples",
                "suggested_tool_input": {
                    "category": "SHIPPING",
                    "n": 5,
                    "offset": 0,
                },
            },
        ],
    )

    assert "[router] structured" in trace
    assert "[tool] resolve_filter_value" in trace
    assert "[reviewer] needs_more" in trace
    assert "[reason] Examples require sample_examples output." in trace
    assert "[suggested_tool] sample_examples" in trace
    assert '"category": "SHIPPING"' in trace
    assert '"n": 5' in trace


def _patch_common_graph_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        agent_profile,
        "read_user_profile_impl",
        lambda user_id: _profile_result(
            user_id,
            "# User Profile\n\n- No durable user facts or preferences have been saved yet.\n",
        ),
    )
    monkeypatch.setattr(
        agent_profile,
        "get_structured_profile_llm",
        lambda: FakeProfileLLM(),
    )


def test_planner_context_includes_recent_structured_results_for_followups() -> None:
    state = graph.create_initial_state(
        query="Show me 3 more.",
        session_id="test_session",
        user_id="max",
    )
    state["route"] = "structured"
    state["route_reason"] = "The user asks for more examples from prior context."
    state["user_profile"] = "# User Profile\n"
    state["last_structured_results"] = [
        {
            "label": "sample_examples",
            "value": 3,
            "query_type": "sample",
            "filters": {
                "category": "REFUND",
                "intent": None,
                "text_query": None,
            },
            "match_count": 6,
        }
    ]

    messages = agent_context._build_planner_messages(
        state=state,
        reviewer_feedback=None,
    )
    context_message = messages[1].content

    assert "Recent structured results:" in context_message
    assert "label=sample_examples" in context_message
    assert "value=3" in context_message
    assert "query_type=sample" in context_message
    assert "REFUND" in context_message


def test_reviewer_context_excludes_previous_structured_results() -> None:
    state = graph.create_initial_state(
       query="Show me 3 more.",
        session_id="test_session",
        user_id="max",
    )
    state["route"] = "structured"
    state["route_reason"] = "The user asks for more examples from prior context."
    state["user_profile"] = "# User Profile\n"
    state["last_structured_results"] = [
        {
            "label": "sample_examples",
            "value": 3,
            "query_type": "sample",
            "filters": {
                "category": "REFUND",
                "intent": None,
                "text_query": None,
            },
            "match_count": 6,
       }
    ]

    messages = agent_context._build_reviewer_messages(state)
    context_message = messages[1].content

    assert "Current turn tool trace:" in context_message
    assert "Evidence boundary: judge only the current turn tool trace." in context_message
    assert "Previous structured" in context_message
    assert "Recent structured results:" not in context_message
    assert "label=sample_examples" not in context_message
    assert "REFUND" not in context_message
    assert "value=3" not in context_message


def test_planner_tool_set_does_not_expose_filter_rows_or_row_id_workflow() -> None:
    assert "filter_rows" not in schemas.VALID_PLANNER_TOOL_NAMES
    assert "count_rows" in schemas.VALID_PLANNER_TOOL_NAMES
    assert "sample_examples" in schemas.VALID_PLANNER_TOOL_NAMES
    assert "group_counts" in schemas.VALID_PLANNER_TOOL_NAMES
    assert "summarize_rows" in schemas.VALID_PLANNER_TOOL_NAMES


@pytest.mark.parametrize(
    ("tool_name", "tool_input"),
    [
        (
            "resolve_filter_value",
            {
                "query": "refund requests",
                "columns": ["category", "intent"],
                "top_k": 0,
            },
        ),
        (
            "sample_examples",
            {
                "category": "REFUND",
                "n": 999,
                "offset": 0,
            },
        ),
        (
            "sample_examples",
            {
                "category": "REFUND",
                "n": 3,
               "offset": -1,
            },
        ),
        (
            "group_counts",
            {
                "group_by": "intent",
                "category": "ACCOUNT",
                "top_k": 100000,
            },
        ),
        (
            "summarize_rows",
            {
                "category": "REFUND",
                "focus": "refund requests",
                "max_examples": 99999,
            },
        ),
    ],
)
def test_tool_executor_rejects_invalid_tool_input_at_boundary(
    tool_name: str,
    tool_input: dict,
) -> None:
    with pytest.raises(ValueError):
        tool_executor._canonical_tool_input(tool_name, tool_input)


def test_structured_query_resolves_filter_then_counts_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        graph,
        "route_query_with_reason",
        lambda query: graph.RouteDecision(
            route="structured",
            reason="The user asks for an exact refund count.",
        ),
    )
    _patch_common_graph_dependencies(monkeypatch)

    monkeypatch.setattr(
        tool_executor,
        "resolve_filter_value_impl",
        lambda query, columns=None, top_k=5: _model_result(
            query=query,
            candidates=[
                {
                    "column": "category",
                    "value": "REFUND",
                    "count": 2992,
                    "score": 1.0,
                    "reason": "Category alias resolves exactly to this dataset value.",
                }
            ],
            recommended_filter={
                "category": "REFUND",
                "intent": None,
            },
            confidence="high",
        ),
    )
    monkeypatch.setattr(
        tool_executor,
        "count_rows_impl",
        lambda category=None, intent=None, text_query=None: _model_result(
            count=2992,
            applied_filters={
                "category": category,
                "intent": intent,
                "text_query": text_query,
            },
        ),
    )

    planner = FakePlannerLLM(
        _plan_call(
            "resolve_filter_value",
            {
                "query": "refund requests",
                "columns": ["category", "intent"],
                "top_k": 5,
            },
        ),
        _plan_call("count_rows", {"category": "REFUND"}),
    )
    reviewer = FakeReviewerLLM(
        _review_needs_more(
            reason="The resolver found the correct dataset filter; now count rows.",
            suggested_tool_name="count_rows",
            suggested_tool_input={"category": "REFUND"},
        ),
        _review_answer("There are 2,992 refund-request rows in the dataset."),
    )

    monkeypatch.setattr(agent_loop, "get_structured_tool_planner_llm", lambda: planner)
    monkeypatch.setattr(agent_loop, "get_structured_observation_reviewer_llm", lambda: reviewer)

    result = graph.invoke_agent(
        query="How many refund requests did we get?",
        session_id="test_session",
        user_id="max",
    )

    assert result["route"] == "structured"
    assert result["final_answer"] == "There are 2,992 refund-request rows in the dataset."
    tool_events = _tool_events(result["tool_trace"])
    reviewer_events = _reviewer_events(result["tool_trace"])
    assert [step["tool_name"] for step in tool_events] == [
        "resolve_filter_value",
        "count_rows",
    ]
    assert tool_events[1]["tool_input"] == {
        "category": "REFUND",
        "intent": None,
        "text_query": None,
    }
    assert [step["reviewer_status"] for step in reviewer_events] == [
        "needs_more",
        "answered",
    ]
    assert reviewer_events[0]["reviewer_reason"] == (
        "The resolver found the correct dataset filter; now count rows."
    )
    assert reviewer_events[0]["suggested_tool_name"] == "count_rows"
    assert reviewer_events[0]["suggested_tool_input"] == {"category": "REFUND"}
    assert reviewer_events[1]["reviewer_final_answer"] == (
        "There are 2,992 refund-request rows in the dataset."
    )
    assert result["last_structured_results"][-1] == {
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
    assert isinstance(result["messages"][-1], AIMessage)


def test_reviewer_suggestion_is_advisory_not_directly_executed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        graph,
        "route_query_with_reason",
        lambda query: graph.RouteDecision(
            route="structured",
            reason="The user asks for an exact refund count.",
        ),
    )
    _patch_common_graph_dependencies(monkeypatch)

    monkeypatch.setattr(
        tool_executor,
        "resolve_filter_value_impl",
        lambda query, columns=None, top_k=5: _model_result(
            query=query,
            candidates=[
                {
                    "column": "category",
                    "value": "REFUND",
                    "count": 2992,
                    "score": 1.0,
                    "reason": "Category alias resolves exactly to this dataset value.",
                }
            ],
            recommended_filter={
                "category": "REFUND",
                "intent": None,
            },
            confidence="high",
        ),
    )
    monkeypatch.setattr(
        tool_executor,
        "count_rows_impl",
        lambda category=None, intent=None, text_query=None: _model_result(
            count=2992,
            applied_filters={
                "category": category,
                "intent": intent,
                "text_query": text_query,
            },
        ),
    )

    def fail_if_sample_examples_called(**kwargs):
        raise AssertionError(
            "Reviewer-suggested sample_examples should not be executed directly."
        )

    monkeypatch.setattr(
        tool_executor,
        "sample_examples_impl",
        fail_if_sample_examples_called,
    )

    planner = FakePlannerLLM(
        _plan_call(
            "resolve_filter_value",
            {
                "query": "refund requests",
                "columns": ["category", "intent"],
                "top_k": 5,
            },
        ),
        _plan_call("count_rows", {"category": "REFUND"}),
    )
    reviewer = FakeReviewerLLM(
        _review_needs_more(
            reason="Intentionally suggest the wrong next tool.",
            suggested_tool_name="sample_examples",
            suggested_tool_input={"category": "REFUND", "n": 3, "offset": 0},
        ),
        _review_answer("There are 2,992 refund-request rows in the dataset."),
    )

    monkeypatch.setattr(agent_loop, "get_structured_tool_planner_llm", lambda: planner)
    monkeypatch.setattr(agent_loop, "get_structured_observation_reviewer_llm", lambda: reviewer)

    result = graph.invoke_agent(
        query="How many refund requests did we get?",
        session_id="test_session",
        user_id="max",
    )

    tool_events = _tool_events(result["tool_trace"])
    assert [step["tool_name"] for step in tool_events] == [
        "resolve_filter_value",
        "count_rows",
    ]
    assert result["final_answer"] == (
        "There are 2,992 refund-request rows in the dataset."
    )

def test_assignment_question_categories_exist_uses_group_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        graph,
        "route_query_with_reason",
        lambda query: graph.RouteDecision(
            route="structured",
            reason="The user asks which categories exist in the dataset.",
        ),
    )
    _patch_common_graph_dependencies(monkeypatch)

    monkeypatch.setattr(
        tool_executor,
        "get_dataset_schema_impl",
        lambda include_sample_values=True: _model_result(
            columns=["row_id", "instruction", "category", "intent", "response"],
            row_count=26872,
            sample_values={
                "category": ["ORDER", "SHIPPING", "CANCEL", "INVOICE", "PAYMENT"],
            },
        ),
    )
    monkeypatch.setattr(
        tool_executor,
        "group_counts_impl",
        lambda group_by, category=None, intent=None, text_query=None, top_k=20: _model_result(
            group_by=group_by,
            counts=[
                {"label": "ORDER", "count": 5000},
                {"label": "SHIPPING", "count": 4200},
                {"label": "ACCOUNT", "count": 3000},
                {"label": "DELIVERY", "count": 2500},
                {"label": "SUBSCRIPTION", "count": 2000},
            ],
            match_count=26872,
            applied_filters={
                "category": category,
                "intent": intent,
                "text_query": text_query,
            },
        ),
    )

    planner = FakePlannerLLM(
        _plan_call("get_dataset_schema", {"include_sample_values": True}),
        _plan_call("group_counts", {"group_by": "category", "top_k": 100}),
    )
    reviewer = FakeReviewerLLM(
        _review_needs_more(
            reason="Schema sample_values are not the complete distinct category set.",
            suggested_tool_name="group_counts",
            suggested_tool_input={"group_by": "category", "top_k": 100},
        ),
        _review_answer(
            "The dataset categories are ORDER, SHIPPING, ACCOUNT, DELIVERY, and SUBSCRIPTION."
        ),
    )

    monkeypatch.setattr(agent_loop, "get_structured_tool_planner_llm", lambda: planner)
    monkeypatch.setattr(agent_loop, "get_structured_observation_reviewer_llm", lambda: reviewer)

    result = graph.invoke_agent(
        query="What categories exist in the dataset?",
        session_id="test_session",
        user_id="max",
    )

    assert result["final_answer"] == (
        "The dataset categories are ORDER, SHIPPING, ACCOUNT, DELIVERY, and SUBSCRIPTION."
    )
    tool_events = _tool_events(result["tool_trace"])
    assert [step["tool_name"] for step in tool_events] == [
        "get_dataset_schema",
        "group_counts",
    ]
    assert tool_events[1]["tool_input"] == {
        "group_by": "category",
        "category": None,
        "intent": None,
        "text_query": None,
        "top_k": 100,
    }


def test_assignment_question_shipping_examples_samples_with_filters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        graph,
        "route_query_with_reason",
        lambda query: graph.RouteDecision(
            route="structured",
            reason="The user asks for examples from a dataset category.",
        ),
    )
    _patch_common_graph_dependencies(monkeypatch)

    monkeypatch.setattr(
        tool_executor,
        "resolve_filter_value_impl",
        lambda query, columns=None, top_k=5: _model_result(
            query=query,
            candidates=[
                {
                    "column": "category",
                    "value": "SHIPPING",
                    "count": 6,
                    "score": 1.0,
                    "reason": "Exact normalized value match.",
                }
            ],
            recommended_filter={
                "category": "SHIPPING",
                "intent": None,
            },
            confidence="high",
        ),
    )
    monkeypatch.setattr(
        tool_executor,
        "sample_examples_impl",
        lambda category=None, intent=None, text_query=None, n=3, offset=0: _model_result(
            examples=[
                _example_row(
                    row_id=20,
                    instruction="Where is my package?",
                    response="You can track it from your account.",
                    category="SHIPPING",
                    intent="track_order",
                )
            ],
            next_offset=5,
            match_count=6,
            applied_filters={
                "category": category,
                "intent": intent,
                "text_query": text_query,
            },
        ),
    )

    planner = FakePlannerLLM(
        _plan_call(
            "resolve_filter_value",
            {
                "query": "SHIPPING",
                "columns": ["category"],
                "top_k": 5,
            },
        ),
        _plan_call(
            "sample_examples",
            {
                "category": "SHIPPING",
                "n": 5,
                "offset": 0,
            },
        ),
    )
    reviewer = FakeReviewerLLM(
        _review_needs_more(
            reason="Examples require sample_examples output.",
            suggested_tool_name="sample_examples",
            suggested_tool_input={
                "category": "SHIPPING",
                "n": 5,
                "offset": 0,
            },
        )
    )

    monkeypatch.setattr(agent_loop, "get_structured_tool_planner_llm", lambda: planner)
    monkeypatch.setattr(agent_loop, "get_structured_observation_reviewer_llm", lambda: reviewer)

    result = graph.invoke_agent(
        query="Show me 5 examples of the SHIPPING category.",
        session_id="test_session",
        user_id="max",
    )

    tool_events = _tool_events(result["tool_trace"])
    assert [step["tool_name"] for step in tool_events] == [
        "resolve_filter_value",
        "sample_examples",
    ]
    assert tool_events[1]["tool_input"] == {
        "category": "SHIPPING",
        "intent": None,
        "text_query": None,
        "n": 5,
        "offset": 0,
    }
    assert result["last_structured_results"][-1] == {
        "label": "sample_examples",
        "value": 5,
        "query_type": "sample",
        "filters": {
            "category": "SHIPPING",
            "intent": None,
            "text_query": None,
        },
        "match_count": 6,
    }
    assert "row_id=20" in result["final_answer"]
    assert "category=SHIPPING" in result["final_answer"]
    reviewer_events = _reviewer_events(result["tool_trace"])
    assert [step["reviewer_status"] for step in reviewer_events] == [
        "needs_more",
        "answered",
    ]
    assert reviewer_events[-1]["suggested_tool_name"] == ""
    assert reviewer_events[-1]["suggested_tool_input"] == {}
    assert "Reviewer LLM skipped" in reviewer_events[-1]["reviewer_reason"]


def test_assignment_question_account_intent_distribution_groups_with_category_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        graph,
        "route_query_with_reason",
        lambda query: graph.RouteDecision(
            route="structured",
            reason="The user asks for intent distribution within a category.",
        ),
    )
    _patch_common_graph_dependencies(monkeypatch)

    monkeypatch.setattr(
        tool_executor,
        "resolve_filter_value_impl",
        lambda query, columns=None, top_k=5: _model_result(
            query=query,
            candidates=[
                {
                    "column": "category",
                    "value": "ACCOUNT",
                    "count": 3,
                    "score": 1.0,
                    "reason": "Exact normalized value match.",
                }
            ],
            recommended_filter={
                "category": "ACCOUNT",
                "intent": None,
            },
            confidence="high",
        ),
    )
    monkeypatch.setattr(
        tool_executor,
        "group_counts_impl",
        lambda group_by, category=None, intent=None, text_query=None, top_k=20: _model_result(
            group_by=group_by,
            counts=[
                {"label": "recover_password", "count": 2},
                {"label": "delete_account", "count": 1},
            ],
            match_count=3,
            applied_filters={
                "category": category,
                "intent": intent,
                "text_query": text_query,
            },
        ),
    )

    planner = FakePlannerLLM(
        _plan_call(
            "resolve_filter_value",
            {
                "query": "ACCOUNT",
                "columns": ["category"],
                "top_k": 5,
            },
        ),
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
        _review_needs_more(
            reason="Intent distribution requires grouping rows with the ACCOUNT category filter.",
            suggested_tool_name="group_counts",
            suggested_tool_input={
                "group_by": "intent",
                "category": "ACCOUNT",
                "top_k": 20,
            },
        ),
        _review_answer(
            "In ACCOUNT, recover_password appears 2 times and delete_account appears once."
        ),
    )

    monkeypatch.setattr(agent_loop, "get_structured_tool_planner_llm", lambda: planner)
    monkeypatch.setattr(agent_loop, "get_structured_observation_reviewer_llm", lambda: reviewer)

    result = graph.invoke_agent(
        query="What is the distribution of intents in the ACCOUNT category?",
        session_id="test_session",
        user_id="max",
    )

    tool_events = _tool_events(result["tool_trace"])
    assert [step["tool_name"] for step in tool_events] == [
        "resolve_filter_value",
        "group_counts",
    ]
    assert tool_events[1]["tool_input"] == {
        "group_by": "intent",
        "category": "ACCOUNT",
        "intent": None,
        "text_query": None,
        "top_k": 20,
    }
    assert result["final_answer"] == (
        "In ACCOUNT, recover_password appears 2 times and delete_account appears once."
    )


@pytest.mark.parametrize(
    "query",
    [
        "Break down ACCOUNT by intent",
        "Intent breakdown for ACCOUNT",
        "What intents appear under ACCOUNT?",
        "Show intents within ACCOUNT",
        "Which intents occur inside ACCOUNT?",
    ],
)
def test_scoped_intent_distribution_phrasings_require_category_filter(
    query: str,
) -> None:
    assert tool_executor._requires_grouped_filtered_scope(
        query=query,
        group_by="intent",
    )


@pytest.mark.parametrize(
    "query",
    [
        "Break down track_refund by category",
        "Category breakdown for track_refund",
        "What categories appear under track_refund?",
        "Show categories within delete_account",
        "Which categories occur inside check_refund_status?",
    ],
)
def test_scoped_category_distribution_phrasings_require_intent_filter(
    query: str,
) -> None:
    assert tool_executor._requires_grouped_filtered_scope(
        query=query,
        group_by="category",
    )


def test_scoped_distribution_answer_contract_blocks_answer_after_only_resolve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        graph,
        "route_query_with_reason",
        lambda query: graph.RouteDecision(
            route="structured",
            reason="The user asks for intent distribution within a category.",
        ),
    )
    _patch_common_graph_dependencies(monkeypatch)

    monkeypatch.setattr(
        tool_executor,
        "resolve_filter_value_impl",
        lambda query, columns=None, top_k=5: _model_result(
            query=query,
            candidates=[
                {
                    "column": "category",
                    "value": "ACCOUNT",
                    "count": 5986,
                    "score": 1.0,
                    "reason": "Category alias resolves exactly to this dataset value.",
                }
            ],
            recommended_filter={
                "category": "ACCOUNT",
                "intent": None,
            },
            confidence="high",
        ),
    )
    monkeypatch.setattr(
        tool_executor,
        "group_counts_impl",
        lambda group_by, category=None, intent=None, text_query=None, top_k=20: _model_result(
            group_by=group_by,
            counts=[
                {"label": "create_account", "count": 997},
                {"label": "delete_account", "count": 995},
            ],
            match_count=5986,
            applied_filters={
                "category": category,
                "intent": intent,
                "text_query": text_query,
            },
        ),
    )

    planner = FakePlannerLLM(
        _plan_call(
            "resolve_filter_value",
            {
                "query": "ACCOUNT",
                "columns": ["category"],
                "top_k": 5,
            },
        ),
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
        _review_answer(
            "Wrong answer that should be blocked because only resolve_filter_value ran."
        ),
        _review_answer(
            "The ACCOUNT category has create_account (997) and delete_account (995)."
        ),
    )

    monkeypatch.setattr(agent_loop, "get_structured_tool_planner_llm", lambda: planner)
    monkeypatch.setattr(agent_loop, "get_structured_observation_reviewer_llm", lambda: reviewer)

    result = graph.invoke_agent(
        query="What is the distribution of intents in the ACCOUNT category?",
        session_id="test_session",
        user_id="max",
    )

    assert result["final_answer"] == (
        "The ACCOUNT category has create_account (997) and delete_account (995)."
    )
    tool_events = _tool_events(result["tool_trace"])
    assert [step["tool_name"] for step in tool_events] == [
        "resolve_filter_value",
        "group_counts",
    ]
    assert tool_events[1]["tool_input"]["category"] == "ACCOUNT"


def test_unstructured_query_summarizes_with_category_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        graph,
        "route_query_with_reason",
        lambda query: graph.RouteDecision(
            route="unstructured",
            reason="The user asks for a qualitative category summary.",
        ),
    )
    _patch_common_graph_dependencies(monkeypatch)

    monkeypatch.setattr(
        tool_executor,
        "resolve_filter_value_impl",
        lambda query, columns=None, top_k=5: _model_result(
            query=query,
            candidates=[
                {
                    "column": "category",
                    "value": "FEEDBACK",
                    "count": 2,
                    "score": 1.0,
                    "reason": "Exact normalized value match.",
                }
            ],
            recommended_filter={
                "category": "FEEDBACK",
                "intent": None,
            },
            confidence="high",
        ),
    )
    monkeypatch.setattr(
        tool_executor,
        "summarize_rows_impl",
        lambda category=None, intent=None, text_query=None, focus="", target_field="both", max_examples=100: _model_result(
            summary="Customers mainly provide product feedback.",
            row_count_used=2,
            match_count=2,
            focus=focus,
            target_field=target_field,
            applied_filters={
                "category": category,
                "intent": intent,
                "text_query": text_query,
            },
        ),
    )

    planner = FakePlannerLLM(
        _plan_call(
            "resolve_filter_value",
            {
                "query": "FEEDBACK",
                "columns": ["category"],
                "top_k": 5,
            },
        ),
        _plan_call(
            "summarize_rows",
            {
                "category": "FEEDBACK",
                "focus": "Summarize the FEEDBACK category.",
                "max_examples": 100,
            },
        ),
    )
    reviewer = FakeReviewerLLM(
        _review_needs_more(
            reason="A qualitative summary requires summarize_rows output.",
            suggested_tool_name="summarize_rows",
            suggested_tool_input={
                "category": "FEEDBACK",
                "focus": "Summarize the FEEDBACK category.",
                "max_examples": 100,
            },
        ),
        _review_answer("Customers mainly provide product feedback."),
    )

    monkeypatch.setattr(agent_loop, "get_structured_tool_planner_llm", lambda: planner)
    monkeypatch.setattr(agent_loop, "get_structured_observation_reviewer_llm", lambda: reviewer)

    result = graph.invoke_agent(
        query="Summarize the FEEDBACK category.",
        session_id="test_session",
        user_id="max",
    )

    tool_events = _tool_events(result["tool_trace"])
    assert [step["tool_name"] for step in tool_events] == [
        "resolve_filter_value",
        "summarize_rows",
    ]
    assert tool_events[1]["tool_input"]["category"] == "FEEDBACK"
    assert result["last_structured_results"][-1]["query_type"] == "summary"
    assert result["last_structured_results"][-1]["value"] == 2
    assert result["final_answer"] == "Customers mainly provide product feedback."


def test_semantic_summary_contract_blocks_text_query_summary_after_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        graph,
        "route_query_with_reason",
        lambda query: graph.RouteDecision(
            route="unstructured",
            reason="The user asks for response patterns for cancellation requests.",
        ),
    )
    _patch_common_graph_dependencies(monkeypatch)

    monkeypatch.setattr(
        tool_executor,
        "resolve_filter_value_impl",
        lambda query, columns=None, top_k=5: _model_result(
            query=query,
            candidates=[
                {
                    "column": "category",
                    "value": "CANCEL",
                    "count": 950,
                    "score": 0.9,
                    "reason": "Dataset value appears inside the user phrase.",
                },
                {
                    "column": "intent",
                    "value": "check_cancellation_fee",
                    "count": 950,
                    "score": 0.6333,
                    "reason": "Token overlap: cancellation.",
                },
            ],
            recommended_filter={
                "category": "CANCEL",
                "intent": None,
            },
            confidence="high",
        ),
    )

    captured_summaries: list[dict] = []

    def fake_summarize_rows(
        category=None,
        intent=None,
        text_query=None,
        focus="",
        target_field="both",
        max_examples=100,
    ):
        captured_summaries.append(
            {
                "category": category,
                "intent": intent,
                "text_query": text_query,
                "focus": focus,
                "target_field": target_field,
                "max_examples": max_examples,
            }
        )

        if text_query == "cancellation requests":
            return _model_result(
                summary="Weak summary from one REFUND policy row.",
                row_count_used=1,
                match_count=1,
                focus=focus,
                target_field=target_field,
                applied_filters={
                    "category": category,
                    "intent": intent,
                    "text_query": text_query,
                },
            )

        return _model_result(
            summary="Agents explain cancellation policy and guide customers through next steps.",
            row_count_used=100,
            match_count=950,
            focus=focus,
            target_field=target_field,
            applied_filters={
                "category": category,
                "intent": intent,
                "text_query": text_query,
            },
        )

    monkeypatch.setattr(tool_executor, "summarize_rows_impl", fake_summarize_rows)

    planner = FakePlannerLLM(
        _plan_call(
            "resolve_filter_value",
            {
                "query": "cancellation requests",
                "columns": ["category", "intent"],
                "top_k": 5,
            },
        ),
        _plan_call(
            "summarize_rows",
            {
                "text_query": "cancellation requests",
                "focus": (
                    "How do customer service representatives typically respond "
                    "to cancellation requests?"
                ),
                "target_field": "response",
                "max_examples": 100,
            },
        ),
        _plan_call(
            "summarize_rows",
            {
                "category": "CANCEL",
                "focus": (
                    "How do customer service representatives typically respond "
                    "to cancellation requests?"
                ),
                "target_field": "response",
                "max_examples": 100,
            },
        ),
    )

    reviewer = FakeReviewerLLM(
        _review_needs_more(
            reason=(
                "The text-query summary lacks resolver evidence for the broad "
                "business phrase."
            ),
            suggested_tool_name="resolve_filter_value",
            suggested_tool_input={
                "query": "cancellation requests",
                "columns": ["category", "intent"],
                "top_k": 5,
            },
        ),
        _review_answer(
            "Wrong premature answer from text-query-only summary."
        ),
        _review_answer(
            "Agents explain cancellation policy and guide customers through next steps."
        ),
    )

    monkeypatch.setattr(agent_loop, "get_structured_tool_planner_llm", lambda: planner)
    monkeypatch.setattr(agent_loop, "get_structured_observation_reviewer_llm", lambda: reviewer)

    result = graph.invoke_agent(
        query="How do customer service representatives typically respond to cancellation requests?",
        session_id="test_session",
        user_id="max",
    )

    tool_events = _tool_events(result["tool_trace"])
    assert [step["tool_name"] for step in tool_events] == [
        "resolve_filter_value",
        "summarize_rows",
        "summarize_rows",
    ]
    assert captured_summaries == [
        {
            "category": None,
            "intent": None,
            "text_query": "cancellation requests",
            "focus": (
                "How do customer service representatives typically respond "
                "to cancellation requests?"
            ),
            "target_field": "response",
            "max_examples": 100,
        },
        {
            "category": "CANCEL",
            "intent": None,
            "text_query": None,
            "focus": (
                "How do customer service representatives typically respond "
                "to cancellation requests?"
            ),
            "target_field": "response",
            "max_examples": 100,
        },
    ]
    assert result["final_answer"] == (
        "Agents explain cancellation policy and guide customers through next steps."
    )


def test_out_of_scope_query_refuses_without_planner(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        graph,
        "route_query_with_reason",
        lambda query: graph.RouteDecision(
            route="out_of_scope",
            reason="The user asks about external facts.",
        ),
    )
    _patch_common_graph_dependencies(monkeypatch)

    def fail_if_planner_called():
        raise AssertionError("Planner should not be called for out-of-scope queries.")

    monkeypatch.setattr(agent_loop, "get_structured_tool_planner_llm", fail_if_planner_called)

    result = graph.invoke_agent(
        query="Who won the World Cup?",
        session_id="test_session",
        user_id="max",
    )

    assert result["route"] == "out_of_scope"
    assert result["final_answer"] == graph.OUT_OF_SCOPE_REFUSAL
    assert result["tool_trace"] == []


@pytest.mark.parametrize(
    "query",
    [
        "What's the best CRM software for handling complaints?",
        "Who is the president of France?",
        "Who won the 2024 Champions League?",
        "Write me a poem about customer service.",
    ],
)
def test_assignment_out_of_scope_questions_refuse_without_planner(
    monkeypatch: pytest.MonkeyPatch,
    query: str,
) -> None:
    monkeypatch.setattr(
        graph,
        "route_query_with_reason",
        lambda user_query: graph.RouteDecision(
            route="out_of_scope",
            reason="The user asks an out-of-scope question.",
        ),
    )
    _patch_common_graph_dependencies(monkeypatch)

    def fail_if_planner_called():
        raise AssertionError("Planner should not be called for out-of-scope queries.")

    monkeypatch.setattr(agent_loop, "get_structured_tool_planner_llm", fail_if_planner_called)

    result = graph.invoke_agent(
        query=query,
        session_id="test_session",
        user_id="max",
    )

    assert result["route"] == "out_of_scope"
    assert result["final_answer"] == graph.OUT_OF_SCOPE_REFUSAL
    assert result["tool_trace"] == []


def test_planner_failure_returns_graceful_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        graph,
        "route_query_with_reason",
        lambda query: graph.RouteDecision(
            route="structured",
            reason="The user asks a dataset question.",
        ),
    )
    _patch_common_graph_dependencies(monkeypatch)
    monkeypatch.setattr(
        agent_loop,
        "get_structured_tool_planner_llm",
        lambda: FailingPlannerLLM(),
    )

    result = graph.invoke_agent(
        query="Analyze the dataset.",
        session_id="test_session",
        user_id="max",
    )

    assert result["final_answer"] == (
        "I could not complete the analysis within the allowed number of "
        "reasoning steps. Please try asking a more specific dataset question."
    )
    assert result["tool_trace"] == []


def test_profile_update_node_saves_durable_observation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        graph,
        "route_query_with_reason",
        lambda query: graph.RouteDecision(
            route="out_of_scope",
            reason="The query is not a dataset question.",
        ),
    )
    monkeypatch.setattr(
        agent_profile,
        "read_user_profile_impl",
        lambda user_id: _profile_result(user_id),
    )
    monkeypatch.setattr(
        agent_profile,
        "get_structured_profile_llm",
        lambda: FakeProfileLLM(
            observation="User prefers file-by-file implementation review."
        ),
    )
    monkeypatch.setattr(
        agent_profile,
        "update_user_profile_impl",
        lambda user_id, new_observation: type(
            "UpdateProfileResult",
            (),
            {
                "user_id": user_id,
                "updated": True,
                "profile": f"# User Profile\n\n- {new_observation}\n",
            },
        )(),
    )

    result = graph.invoke_agent(
        query="I prefer file-by-file implementation review.",
        session_id="test_session",
        user_id="max",
    )

    assert "- User prefers file-by-file implementation review." in result["user_profile"]


def test_build_graph_config_uses_session_id_as_thread_id() -> None:
    config = graph._build_graph_config(
        session_id="demo_session",
        user_id="max",
        max_iterations=12,
    )

    assert config["configurable"]["thread_id"] == "demo_session"
    assert config["configurable"]["user_id"] == "max"
    assert config["recursion_limit"] == 17


def test_create_invocation_state_returns_full_state_for_new_thread() -> None:
    class EmptyCheckpointState:
        values = {}

    class FakeGraph:
        def get_state(self, config):
            return EmptyCheckpointState()

    config = graph._build_graph_config(
        session_id="new_session",
        user_id="max",
        max_iterations=12,
    )

    result = graph._create_invocation_state(
        graph=FakeGraph(),
        query="How many refund requests?",
        session_id="new_session",
        user_id="max",
        max_iterations=12,
        config=config,
    )

    assert result["session_id"] == "new_session"
    assert result["user_id"] == "max"
    assert result["last_structured_results"] == []
    assert isinstance(result["messages"][0], HumanMessage)
    assert result["messages"][0].content == "How many refund requests?"


def test_create_invocation_state_returns_partial_update_for_existing_thread() -> None:
    class ExistingCheckpointState:
        values = {
            "messages": [HumanMessage(content="Previous question")],
            "last_structured_results": [
                {
                    "label": "count_rows",
                    "value": 3,
                    "query_type": "count",
                    "filters": {
                        "category": "REFUND",
                        "intent": None,
                        "text_query": None,
                    },
                    "match_count": 3,
                }
            ],
        }

    class FakeGraph:
        def get_state(self, config):
            return ExistingCheckpointState()

    config = graph._build_graph_config(
        session_id="existing_session",
        user_id="max",
        max_iterations=12,
    )

    result = graph._create_invocation_state(
        graph=FakeGraph(),
        query="Show me 3 more.",
        session_id="existing_session",
        user_id="max",
        max_iterations=12,
        config=config,
    )

    assert result["session_id"] == "existing_session"
    assert result["user_id"] == "max"
    assert result["tool_trace"] == []
    assert result["final_answer"] is None
    assert "last_structured_results" not in result
    assert isinstance(result["messages"][0], HumanMessage)
    assert result["messages"][0].content == "Show me 3 more."


def test_load_user_profile_node_returns_partial_state_update(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        agent_profile,
        "read_user_profile_impl",
        lambda user_id: _profile_result(
            user_id,
            "# User Profile\n\n- Test profile\n",
        ),
    )

    state = graph.create_initial_state(
        query="What do you remember about me?",
        session_id="test_session",
        user_id="max",
    )

    result = graph.load_user_profile_node(state)

    assert result == {
        "user_profile": "# User Profile\n\n- Test profile\n",
    }


def test_router_node_returns_partial_state_update(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        graph,
        "route_query_with_reason",
        lambda query: graph.RouteDecision(
            route="structured",
            reason="The query asks for an exact dataset count.",
        ),
    )

    state = graph.create_initial_state(
        query="How many refund requests?",
        session_id="test_session",
        user_id="max",
    )

    result = graph.router_node(state)

    assert result == {
        "route": "structured",
        "route_reason": "The query asks for an exact dataset count.",
    }


def test_route_after_router_sends_dataset_queries_to_data_agent_loop() -> None:
    state = graph.create_initial_state(
        query="How many refund requests?",
        session_id="test_session",
        user_id="max",
    )
    state["route"] = "structured"

    assert graph.route_after_router(state) == "data_agent_loop_node"

    state["route"] = "unstructured"

    assert graph.route_after_router(state) == "data_agent_loop_node"


def test_route_after_router_sends_out_of_scope_to_refusal_node() -> None:
    state = graph.create_initial_state(
        query="Who won the World Cup?",
        session_id="test_session",
        user_id="max",
    )
    state["route"] = "out_of_scope"

    assert graph.route_after_router(state) == "refusal_node"


def test_follow_up_show_me_three_more_uses_previous_filters_and_offset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = graph.create_initial_state(
        query="Show me 3 more.",
        session_id="test_session",
        user_id="max",
    )
    state["route"] = "structured"
    state["route_reason"] = "The user asks for more examples from the previous subset."
    state["user_profile"] = "# User Profile\n"
    state["last_structured_results"] = [
        {
            "label": "sample_examples",
            "value": 3,
            "query_type": "sample",
            "filters": {
                "category": "REFUND",
                "intent": None,
                "text_query": None,
            },
            "match_count": 6,
        }
    ]

    captured_inputs: list[dict] = []

    monkeypatch.setattr(
        tool_executor,
        "sample_examples_impl",
        lambda category=None, intent=None, text_query=None, n=3, offset=0: (
            captured_inputs.append(
                {
                    "category": category,
                    "intent": intent,
                    "text_query": text_query,
                    "n": n,
                    "offset": offset,
                }
            )
            or _model_result(
                examples=[
                    _example_row(
                        row_id=13,
                        category="REFUND",
                        intent="check_refund_status",
                        instruction="Example 4",
                        response="Refunds usually take several business days.",
                    ),
                    _example_row(
                        row_id=14,
                        category="REFUND",
                        intent="get_refund",
                        instruction="Example 5",
                        response="You can request a refund through your account.",
                    ),
                    _example_row(
                        row_id=15,
                        category="REFUND",
                        intent="check_refund_status",
                        instruction="Example 6",
                        response="Your refund status is available in your account.",
                    ),
                ],
                next_offset=6,
                match_count=6,
                applied_filters={
                    "category": category,
                    "intent": intent,
                    "text_query": text_query,
                },
            )
        ),
    )

    def fail_if_planner_called():
        raise AssertionError("Planner should not be called for deterministic follow-up examples.")

    monkeypatch.setattr(agent_loop, "get_structured_tool_planner_llm", fail_if_planner_called)

    result = graph.data_agent_loop_node(state)

    assert captured_inputs == [
        {
            "category": "REFUND",
            "intent": None,
            "text_query": None,
            "n": 3,
            "offset": 3,
        }
    ]
    assert "row_id=13" in result["final_answer"]
    assert "row_id=14" in result["final_answer"]
    assert "row_id=15" in result["final_answer"]
    assert _tool_events(result["tool_trace"])[-1]["tool_name"] == "sample_examples"
    reviewer_events = _reviewer_events(result["tool_trace"])
    assert len(reviewer_events) == 1
    assert reviewer_events[0]["reviewer_status"] == "answered"
    assert reviewer_events[0]["suggested_tool_name"] == ""
    assert reviewer_events[0]["suggested_tool_input"] == {}
    assert "Planner and reviewer LLMs skipped" in reviewer_events[0]["reviewer_reason"]
    assert state["last_structured_results"][-1] == {
        "label": "sample_examples",
        "value": 6,
        "query_type": "sample",
        "filters": {
            "category": "REFUND",
            "intent": None,
            "text_query": None,
        },
        "match_count": 6,
    }


def test_sample_formatter_returns_applied_filters_for_follow_up_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        tool_executor,
        "sample_examples_impl",
        lambda category=None, intent=None, text_query=None, n=3, offset=0: _model_result(
            examples=[
                _example_row(
                    row_id=1,
                    category="REFUND",
                    intent="get_refund",
                    instruction="I want my money back.",
                    response="You can request a refund from your account.",
                )
            ],
            next_offset=1,
            match_count=2,
            applied_filters={
                "category": "REFUND",
                "intent": None,
                "text_query": None,
            },
        ),
    )

    observation, next_offset, match_count, applied_filters = (
        tool_executor._format_sample_examples_observation(
            filters={
                "category": "money back",
                "intent": None,
                "text_query": None,
            },
            n=1,
            offset=0,
        )
    )

    assert "row_id=1" in observation
    assert next_offset == 1
    assert match_count == 2
    assert applied_filters == {
        "category": "REFUND",
        "intent": None,
        "text_query": None,
    }


def test_more_examples_detector_requires_example_context_for_ambiguous_next_counts() -> None:
    assert agent_followups._is_more_examples_query("next 2") is False
    assert agent_followups._is_more_examples_query("another 2") is False

    assert agent_followups._is_more_examples_query("next 2 rows") is True
    assert agent_followups._is_more_examples_query("another 2 examples") is True
    assert agent_followups._is_more_examples_query("show me 3 more") is True
    assert agent_followups._is_more_examples_query("give me 2 more cases") is True


def test_total_of_last_two_follow_up_goes_to_planner_with_filter_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = graph.create_initial_state(
        query="What is the total count of the last two?",
        session_id="test_session",
        user_id="max",
    )
    state["route"] = "structured"
    state["route_reason"] = "The user asks to combine recent count results."
    state["user_profile"] = "# User Profile\n"
    state["last_structured_results"] = [
        {
            "label": "complaints",
            "value": 514,
            "query_type": "count",
            "filters": {
                "category": "COMPLAINT",
                "intent": None,
                "text_query": None,
            },
            "match_count": 514,
        },
        {
            "label": "refunds",
            "value": 842,
            "query_type": "count",
            "filters": {
                "category": "REFUND",
                "intent": None,
                "text_query": None,
            },
            "match_count": 842,
        },
    ]

    planner = FakePlannerLLM(
        _plan_final("The total count of the last two results is 1,356.")
    )

    monkeypatch.setattr(agent_loop, "get_structured_tool_planner_llm", lambda: planner)

    result = graph.data_agent_loop_node(state)

    context_message = planner.received_messages[0][1].content

    assert "complaints" in context_message
    assert "514" in context_message
    assert "COMPLAINT" in context_message
    assert "refunds" in context_message
    assert "842" in context_message
    assert "REFUND" in context_message
    assert "row_ids" not in context_message
    assert result["final_answer"] == "The total count of the last two results is 1,356."


def test_create_invocation_state_falls_back_to_initial_state_when_checkpoint_read_fails() -> None:
    class FailingCheckpointGraph:
        def get_state(self, config):
            raise RuntimeError("Checkpoint unavailable.")

    config = graph._build_graph_config(
        session_id="test_session",
        user_id="max",
        max_iterations=12,
    )

    result = graph._create_invocation_state(
        graph=FailingCheckpointGraph(),
        query="How many refund requests?",
        session_id="test_session",
        user_id="max",
        max_iterations=12,
        config=config,
    )

    assert result["session_id"] == "test_session"
    assert result["user_id"] == "max"
    assert result["last_structured_results"] == []
    assert isinstance(result["messages"][0], HumanMessage)


def test_unknown_tool_error_returns_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        graph,
        "route_query_with_reason",
        lambda query: graph.RouteDecision(
            route="structured",
            reason="The user asks a dataset question.",
        ),
    )
    _patch_common_graph_dependencies(monkeypatch)

    planner = FakePlannerLLM(
        schemas.ToolPlanDecision(
            action="call_tool",
            tool_name="",
            tool_input={},
            reason="Malformed planner output.",
        )
    )

    monkeypatch.setattr(agent_loop, "get_structured_tool_planner_llm", lambda: planner)

    result = graph.invoke_agent(
        query="How many rows are there?",
        session_id="test_session",
        user_id="max",
    )

    assert result["final_answer"] == (
        "I could not complete the analysis within the allowed number of "
        "reasoning steps. Please try asking a more specific dataset question."
    )


