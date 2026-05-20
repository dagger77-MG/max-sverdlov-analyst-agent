from __future__ import annotations

import json
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app import graph


@pytest.fixture(autouse=True)
def patch_checkpointer(monkeypatch: pytest.MonkeyPatch):
    """Keep graph unit tests independent from optional SQLite checkpoints."""
    graph.build_graph.cache_clear()

    if hasattr(graph.get_checkpointer, "cache_clear"):
        graph.get_checkpointer.cache_clear()

    if hasattr(graph.get_structured_tool_planner_llm, "cache_clear"):
        graph.get_structured_tool_planner_llm.cache_clear()

    if hasattr(graph.get_structured_observation_reviewer_llm, "cache_clear"):
        graph.get_structured_observation_reviewer_llm.cache_clear()

    if hasattr(graph.get_structured_profile_llm, "cache_clear"):
        graph.get_structured_profile_llm.cache_clear()

    monkeypatch.setattr(graph, "get_checkpointer", lambda: None)

    yield

    graph.build_graph.cache_clear()


class FakeProfileLLM:
    def __init__(self, observation: str = "") -> None:
        self.observation = observation

    def invoke(self, messages):
        return graph.ProfileObservationDecision(observation=self.observation)


class FakePlannerLLM:
    def __init__(self, *decisions: graph.ToolPlanDecision) -> None:
        self.decisions = list(decisions)
        self.received_messages = []

    def invoke(self, messages):
        self.received_messages.append(messages)
        if not self.decisions:
            raise AssertionError("Planner received more calls than expected.")
        return self.decisions.pop(0)


class FakeReviewerLLM:
    def __init__(self, *decisions: graph.ObservationReviewDecision) -> None:
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


def _plan_call(tool_name: str, tool_input: dict) -> graph.ToolPlanDecision:
    return graph.ToolPlanDecision(
        action="call_tool",
        tool_name=tool_name,
        tool_input=tool_input,
        reason=f"Call {tool_name}.",
    )


def _plan_final(answer: str) -> graph.ToolPlanDecision:
    return graph.ToolPlanDecision(
        action="final_answer",
        final_answer=answer,
        reason="The available context is sufficient.",
    )


def _review_answer(answer: str) -> graph.ObservationReviewDecision:
    return graph.ObservationReviewDecision(
        status="answered",
        reason="The observations are sufficient.",
        final_answer=answer,
    )


def _review_needs_more(
    reason: str,
    suggested_tool_name: str,
    suggested_tool_input: dict,
) -> graph.ObservationReviewDecision:
    return graph.ObservationReviewDecision(
        status="needs_more",
        reason=reason,
        suggested_tool_name=suggested_tool_name,
        suggested_tool_input=suggested_tool_input,
    )


def _patch_common_graph_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        graph,
        "read_user_profile_impl",
        lambda user_id: _profile_result(
            user_id,
            "# User Profile\n\n- No durable user facts or preferences have been saved yet.\n",
        ),
    )
    monkeypatch.setattr(
        graph,
        "get_structured_profile_llm",
        lambda: FakeProfileLLM(),
    )


def test_structured_query_uses_planner_executor_reviewer_loop(
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
        graph,
        "filter_rows_impl",
        lambda category=None, intent=None, text_query=None, limit=None: _model_result(
            row_ids=[1, 2, 3],
            match_count=3,
            applied_filters={
                "category": category,
                "intent": intent,
                "text_query": text_query,
                "limit": limit,
            },
        ),
    )

    planner = FakePlannerLLM(_plan_call("filter_rows", {"category": "REFUND"}))
    reviewer = FakeReviewerLLM(_review_answer("There are 3 refund rows in the dataset."))

    monkeypatch.setattr(graph, "get_structured_tool_planner_llm", lambda: planner)
    monkeypatch.setattr(graph, "get_structured_observation_reviewer_llm", lambda: reviewer)

    result = graph.invoke_agent(
        query="How many refund requests did we get?",
        session_id="test_session",
        user_id="max",
    )

    assert result["route"] == "structured"
    assert result["final_answer"] == "There are 3 refund rows in the dataset."
    assert [step["tool_name"] for step in result["tool_trace"]] == ["filter_rows"]
    assert result["tool_trace"][0]["tool_input"] == {"category": "REFUND"}
    assert result["last_structured_results"][-1] == {
        "label": str(
            {
                "category": "REFUND",
                "intent": None,
                "text_query": None,
                "limit": None,
            }
        ),
        "value": 3,
        "query_type": "filter",
        "row_ids": [1, 2, 3],
    }
    assert isinstance(result["messages"][-1], AIMessage)


def test_refund_count_resolves_filter_value_before_filtering(
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
        graph,
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
        graph,
        "filter_rows_impl",
        lambda category=None, intent=None, text_query=None, limit=None: _model_result(
            row_ids=[5917, 5918, 5919],
            match_count=2992,
            applied_filters={
                "category": category,
                "intent": intent,
                "text_query": text_query,
                "limit": limit,
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
        _plan_call("filter_rows", {"category": "REFUND"}),
    )
    reviewer = FakeReviewerLLM(
        _review_needs_more(
            reason="The resolver found the correct dataset filter; now filter rows.",
            suggested_tool_name="filter_rows",
            suggested_tool_input={"category": "REFUND"},
        ),
        _review_answer("There are 2,992 refund-request rows in the dataset."),
    )

    monkeypatch.setattr(graph, "get_structured_tool_planner_llm", lambda: planner)
    monkeypatch.setattr(graph, "get_structured_observation_reviewer_llm", lambda: reviewer)

    result = graph.invoke_agent(
        query="How many refund requests did we get?",
        session_id="test_session",
        user_id="max",
    )

    assert result["final_answer"] == "There are 2,992 refund-request rows in the dataset."
    assert [step["tool_name"] for step in result["tool_trace"]] == [
        "resolve_filter_value",
        "filter_rows",
    ]
    assert result["tool_trace"][0]["tool_input"] == {
        "query": "refund requests",
        "columns": ["category", "intent"],
        "top_k": 5,
    }
    assert result["tool_trace"][1]["tool_input"] == {"category": "REFUND"}
    assert result["last_structured_results"][-1]["value"] == 2992
    assert result["last_structured_results"][-1]["query_type"] == "filter"


def test_assignment_question_categories_exist_reviewer_rejects_schema_sample_then_uses_group_counts(
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
        graph,
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
        graph,
        "group_counts_impl",
        lambda group_by, row_ids=None, top_k=20: _model_result(
            group_by=group_by,
            counts=[
                {"label": "ORDER", "count": 5000},
                {"label": "SHIPPING", "count": 4200},
                {"label": "ACCOUNT", "count": 3000},
                {"label": "DELIVERY", "count": 2500},
                {"label": "SUBSCRIPTION", "count": 2000},
            ],
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

    monkeypatch.setattr(graph, "get_structured_tool_planner_llm", lambda: planner)
    monkeypatch.setattr(graph, "get_structured_observation_reviewer_llm", lambda: reviewer)

    result = graph.invoke_agent(
        query="What categories exist in the dataset?",
        session_id="test_session",
        user_id="max",
    )

    assert result["route"] == "structured"
    assert result["final_answer"] == (
        "The dataset categories are ORDER, SHIPPING, ACCOUNT, DELIVERY, and SUBSCRIPTION."
    )
    assert [step["tool_name"] for step in result["tool_trace"]] == [
        "get_dataset_schema",
        "group_counts",
    ]
    assert result["tool_trace"][1]["tool_input"] == {
        "group_by": "category",
        "top_k": 100,
    }


def test_assignment_question_shipping_examples_filters_then_samples(
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
        graph,
        "filter_rows_impl",
        lambda category=None, intent=None, text_query=None, limit=None: _model_result(
            row_ids=[20, 21, 22, 23, 24, 25],
            match_count=6,
            applied_filters={
                "category": category,
                "intent": intent,
                "text_query": text_query,
                "limit": limit,
            },
        ),
    )

    monkeypatch.setattr(
        graph,
        "sample_examples_impl",
        lambda row_ids=None, n=3, offset=0: type(
            "SampleExamplesResult",
            (),
            {
                "examples": [
                    type(
                        "ExampleRow",
                        (),
                        {
                            "row_id": 20,
                            "instruction": "Where is my package?",
                            "response": "You can track it from your account.",
                            "category": "SHIPPING",
                            "intent": "track_order",
                        },
                    )()
                ],
                "next_offset": 5,
            },
        )(),
    )

    planner = FakePlannerLLM(
        _plan_call("filter_rows", {"category": "SHIPPING"}),
        _plan_call(
            "sample_examples",
            {"row_ids": [20, 21, 22, 23, 24, 25], "n": 5, "offset": 0},
        ),
    )
    reviewer = FakeReviewerLLM(
        _review_needs_more(
            reason="Examples require sample_examples output.",
            suggested_tool_name="sample_examples",
            suggested_tool_input={
                "row_ids": [20, 21, 22, 23, 24, 25],
                "n": 5,
                "offset": 0,
            },
        ),
        _review_answer("Here are 5 SHIPPING examples from the dataset."),
    )

    monkeypatch.setattr(graph, "get_structured_tool_planner_llm", lambda: planner)
    monkeypatch.setattr(graph, "get_structured_observation_reviewer_llm", lambda: reviewer)

    result = graph.invoke_agent(
        query="Show me 5 examples of the SHIPPING category.",
        session_id="test_session",
        user_id="max",
    )

    assert result["route"] == "structured"
    assert [step["tool_name"] for step in result["tool_trace"]] == [
        "filter_rows",
        "sample_examples",
    ]
    assert result["tool_trace"][0]["tool_input"] == {"category": "SHIPPING"}
    assert result["tool_trace"][1]["tool_input"] == {
        "row_ids": [20, 21, 22, 23, 24, 25],
        "n": 5,
        "offset": 0,
    }
    assert result["last_structured_results"][-1] == {
        "label": "sample_examples",
        "value": 5,
        "query_type": "sample",
        "row_ids": [20, 21, 22, 23, 24, 25],
    }


def test_assignment_question_account_intent_distribution_filters_then_groups(
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
        graph,
        "filter_rows_impl",
        lambda category=None, intent=None, text_query=None, limit=None: _model_result(
            row_ids=[30, 31, 32],
            match_count=3,
            applied_filters={
                "category": category,
                "intent": intent,
                "text_query": text_query,
                "limit": limit,
            },
        ),
    )
    monkeypatch.setattr(
        graph,
        "group_counts_impl",
        lambda group_by, row_ids=None, top_k=20: _model_result(
            group_by=group_by,
            counts=[
                {"label": "recover_password", "count": 2},
                {"label": "delete_account", "count": 1},
            ],
        ),
    )

    planner = FakePlannerLLM(
        _plan_call("filter_rows", {"category": "ACCOUNT"}),
        _plan_call(
            "group_counts",
            {
                "group_by": "intent",
                "row_ids": [30, 31, 32],
                "top_k": 20,
            },
        ),
    )
    reviewer = FakeReviewerLLM(
        _review_needs_more(
            reason="Intent distribution requires grouping filtered ACCOUNT rows.",
            suggested_tool_name="group_counts",
            suggested_tool_input={
                "group_by": "intent",
                "row_ids": [30, 31, 32],
                "top_k": 20,
            },
        ),
        _review_answer(
            "In ACCOUNT, recover_password appears 2 times and delete_account appears once."
        ),
    )

    monkeypatch.setattr(graph, "get_structured_tool_planner_llm", lambda: planner)
    monkeypatch.setattr(graph, "get_structured_observation_reviewer_llm", lambda: reviewer)

    result = graph.invoke_agent(
        query="What is the distribution of intents in the ACCOUNT category?",
        session_id="test_session",
        user_id="max",
    )

    assert result["route"] == "structured"
    assert [step["tool_name"] for step in result["tool_trace"]] == [
        "filter_rows",
        "group_counts",
    ]
    assert result["tool_trace"][0]["tool_input"] == {"category": "ACCOUNT"}
    assert result["tool_trace"][1]["tool_input"] == {
        "group_by": "intent",
        "row_ids": [30, 31, 32],
        "top_k": 20,
    }


def test_group_counts_rejects_symbolic_row_ids_without_calling_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = graph.create_initial_state(
        query="What is the distribution of intents in the ACCOUNT category?",
        session_id="test_session",
        user_id="max",
    )

    def fail_if_group_counts_called(group_by, row_ids=None, top_k=20):
        raise AssertionError(
            "group_counts_impl should not be called with invalid row_ids."
        )

    monkeypatch.setattr(graph, "group_counts_impl", fail_if_group_counts_called)

    graph._execute_selected_tool(
        state=state,
        tool_name="group_counts",
        tool_input={
            "group_by": "intent",
            "row_ids": "resolve_filter_value",
            "query": "ACCOUNT",
            "columns": ["category"],
            "top_k": 5,
        },
    )

    assert len(state["tool_trace"]) == 1
    assert state["tool_trace"][0]["tool_name"] == "group_counts"

    observation = json.loads(state["tool_trace"][0]["observation"])
    assert "error" in observation
    assert "row_ids must be a list of integer row IDs or null" in observation["error"]
    assert "required_next_step" in observation
    assert state["last_structured_results"] == []


def test_group_counts_rejects_global_grouping_for_scoped_distribution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = graph.create_initial_state(
        query="What is the distribution of intents in the ACCOUNT category?",
        session_id="test_session",
        user_id="max",
    )

    def fail_if_group_counts_called(group_by, row_ids=None, top_k=20):
        raise AssertionError(
            "group_counts_impl should not be called for an unscoped distribution."
        )

    monkeypatch.setattr(graph, "group_counts_impl", fail_if_group_counts_called)

    graph._execute_selected_tool(
        state=state,
        tool_name="group_counts",
        tool_input={
            "group_by": "intent",
            "top_k": 5,
        },
    )

    assert len(state["tool_trace"]) == 1
    assert state["tool_trace"][0]["tool_name"] == "group_counts"

    observation = json.loads(state["tool_trace"][0]["observation"])
    assert "error" in observation
    assert "would group all rows" in observation["error"]
    assert "filter_rows" in observation["required_next_step"]
    assert state["last_structured_results"] == []


def test_scoped_distribution_answer_contract_blocks_answer_after_late_resolve(
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
        graph,
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
        graph,
        "filter_rows_impl",
        lambda category=None, intent=None, text_query=None, limit=None: _model_result(
            row_ids=[30, 31, 32, 33, 34, 35],
            match_count=5986,
            applied_filters={
                "category": category,
                "intent": intent,
                "text_query": text_query,
                "limit": limit,
            },
        ),
    )
    monkeypatch.setattr(
        graph,
        "group_counts_impl",
        lambda group_by, row_ids=None, top_k=20: _model_result(
            group_by=group_by,
            counts=[
                {"label": "create_account", "count": 997},
                {"label": "delete_account", "count": 995},
                {"label": "edit_account", "count": 1000},
                {"label": "recover_password", "count": 995},
                {"label": "registration_problems", "count": 999},
                {"label": "switch_account", "count": 1000},
            ],
        ),
    )

    planner = FakePlannerLLM(
        _plan_call(
            "group_counts",
            {
                "group_by": "intent",
                "row_ids": "resolve_filter_value",
                "query": "ACCOUNT",
                "columns": ["category"],
                "top_k": 5,
            },
        ),
        _plan_call(
            "resolve_filter_value",
            {
                "query": "ACCOUNT",
                "columns": ["category"],
                "top_k": 5,
            },
        ),
        _plan_call("filter_rows", {"category": "ACCOUNT"}),
        _plan_call(
            "group_counts",
            {
                "group_by": "intent",
                "scope": "latest_filter",
                "top_k": 20,
            },
        ),
    )
    reviewer = FakeReviewerLLM(
        _review_needs_more(
            reason="The symbolic row_ids group_counts call is invalid.",
            suggested_tool_name="resolve_filter_value",
            suggested_tool_input={
                "query": "ACCOUNT",
                "columns": ["category"],
                "top_k": 5,
            },
        ),
        _review_answer(
            "Wrong answer that should be blocked because only resolve_filter_value ran."
        ),
        _review_needs_more(
            reason="The ACCOUNT subset exists; now group intents over that subset.",
            suggested_tool_name="group_counts",
            suggested_tool_input={
                "group_by": "intent",
                "scope": "latest_filter",
                "top_k": 20,
            },
        ),
        _review_answer(
            "The ACCOUNT category has create_account (997), delete_account (995), "
            "edit_account (1000), recover_password (995), registration_problems "
            "(999), and switch_account (1000)."
        ),
    )

    monkeypatch.setattr(graph, "get_structured_tool_planner_llm", lambda: planner)
    monkeypatch.setattr(graph, "get_structured_observation_reviewer_llm", lambda: reviewer)

    result = graph.invoke_agent(
        query="What is the distribution of intents in the ACCOUNT category?",
        session_id="test_session",
        user_id="max",
    )

    assert result["final_answer"] == (
        "The ACCOUNT category has create_account (997), delete_account (995), "
        "edit_account (1000), recover_password (995), registration_problems "
        "(999), and switch_account (1000)."
    )
    assert [step["tool_name"] for step in result["tool_trace"]] == [
        "group_counts",
        "resolve_filter_value",
        "filter_rows",
        "group_counts",
    ]

    bad_group_observation = json.loads(result["tool_trace"][0]["observation"])
    assert "error" in bad_group_observation
    assert result["tool_trace"][3]["tool_input"] == {
        "group_by": "intent",
        "scope": "latest_filter",
        "top_k": 20,
        "row_ids": [30, 31, 32, 33, 34, 35],
    }


def test_unstructured_query_filters_then_summarizes(
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
        graph,
        "filter_rows_impl",
        lambda category=None, intent=None, text_query=None, limit=None: _model_result(
            row_ids=[10, 11],
            match_count=2,
            applied_filters={
                "category": category,
                "intent": intent,
                "text_query": text_query,
                "limit": limit,
            },
        ),
    )
    monkeypatch.setattr(
        graph,
        "summarize_rows_impl",
        lambda row_ids, focus, max_examples=100: _model_result(
            summary="Customers mainly provide product feedback.",
            row_count_used=2,
            focus=focus,
        ),
    )

    planner = FakePlannerLLM(
        _plan_call("filter_rows", {"category": "FEEDBACK"}),
        _plan_call(
            "summarize_rows",
            {
                "row_ids": [10, 11],
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
                "row_ids": [10, 11],
                "focus": "Summarize the FEEDBACK category.",
                "max_examples": 100,
            },
        ),
        _review_answer("Customers mainly provide product feedback."),
    )

    monkeypatch.setattr(graph, "get_structured_tool_planner_llm", lambda: planner)
    monkeypatch.setattr(graph, "get_structured_observation_reviewer_llm", lambda: reviewer)

    result = graph.invoke_agent(
        query="Summarize the FEEDBACK category.",
        session_id="test_session",
        user_id="max",
    )

    assert result["route"] == "unstructured"
    assert result["final_answer"] == "Customers mainly provide product feedback."
    assert [step["tool_name"] for step in result["tool_trace"]] == [
        "filter_rows",
        "summarize_rows",
    ]
    assert result["last_structured_results"][-1]["query_type"] == "filter"
    assert result["last_structured_results"][-1]["value"] == 2


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

    monkeypatch.setattr(graph, "get_structured_tool_planner_llm", fail_if_planner_called)

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

    monkeypatch.setattr(graph, "get_structured_tool_planner_llm", fail_if_planner_called)

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
        graph,
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
        graph,
        "read_user_profile_impl",
        lambda user_id: _profile_result(user_id),
    )
    monkeypatch.setattr(
        graph,
        "get_structured_profile_llm",
        lambda: FakeProfileLLM(
            observation="User prefers file-by-file implementation review."
        ),
    )
    monkeypatch.setattr(
        graph,
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
                    "row_ids": [1, 2, 3],
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
        graph,
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


def test_follow_up_show_me_three_more_uses_previous_row_ids_and_offset(
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
            "row_ids": [10, 11, 12, 13, 14, 15],
        }
    ]

    captured_inputs: list[dict] = []

    monkeypatch.setattr(
        graph,
        "sample_examples_impl",
        lambda row_ids=None, n=3, offset=0: (
            captured_inputs.append(
                {
                    "row_ids": row_ids,
                    "n": n,
                    "offset": offset,
                }
            )
            or type(
                "SampleExamplesResult",
                (),
                {
                    "examples": [
                        type(
                            "ExampleRow",
                            (),
                            {
                                "row_id": 13,
                                "category": "REFUND",
                                "intent": "check_refund_status",
                                "instruction": "Example 4",
                                "response": "Refunds usually take several business days.",
                            },
                        )(),
                        type(
                            "ExampleRow",
                            (),
                            {
                                "row_id": 14,
                                "category": "REFUND",
                                "intent": "get_refund",
                                "instruction": "Example 5",
                                "response": "You can request a refund through your account.",
                            },
                        )(),
                        type(
                            "ExampleRow",
                            (),
                            {
                                "row_id": 15,
                                "category": "REFUND",
                                "intent": "check_refund_status",
                                "instruction": "Example 6",
                                "response": "Your refund status is available in your account.",
                            },
                        )(),
                    ],
                    "next_offset": 6,
                },
            )()
        ),
    )

    def fail_if_planner_called():
        raise AssertionError("Planner should not be called for deterministic follow-up examples.")

    monkeypatch.setattr(graph, "get_structured_tool_planner_llm", fail_if_planner_called)

    result = graph.data_agent_loop_node(state)

    assert captured_inputs == [
        {
            "row_ids": [10, 11, 12, 13, 14, 15],
            "n": 3,
            "offset": 3,
        }
    ]
    assert "row_id=13" in result["final_answer"]
    assert "row_id=14" in result["final_answer"]
    assert "row_id=15" in result["final_answer"]
    assert result["tool_trace"][-1]["tool_name"] == "sample_examples"
    assert state["last_structured_results"][-1] == {
        "label": "sample_examples",
        "value": 6,
        "query_type": "sample",
        "row_ids": [10, 11, 12, 13, 14, 15],
    }


def test_total_of_last_two_follow_up_goes_to_planner_with_context(
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
            "row_ids": None,
        },
        {
            "label": "refunds",
            "value": 842,
            "query_type": "filter",
            "row_ids": [1, 2, 3],
        },
    ]

    planner = FakePlannerLLM(
        _plan_final("The total count of the last two results is 1,356.")
    )

    monkeypatch.setattr(graph, "get_structured_tool_planner_llm", lambda: planner)

    result = graph.data_agent_loop_node(state)

    context_message = planner.received_messages[0][1].content

    assert "complaints" in context_message
    assert "514" in context_message
    assert "refunds" in context_message
    assert "842" in context_message
    assert result["final_answer"] == "The total count of the last two results is 1,356."


def test_sample_examples_trace_includes_full_example_details(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        graph,
        "sample_examples_impl",
        lambda row_ids=None, n=3, offset=0: type(
            "SampleExamplesResult",
            (),
            {
                "examples": [
                    type(
                        "ExampleRow",
                        (),
                        {
                            "row_id": 10,
                            "category": "REFUND",
                            "intent": "check_refund_status",
                            "instruction": "Where is my refund?",
                            "response": "You can check your refund status in your account.",
                        },
                    )()
                ],
                "next_offset": 1,
            },
        )(),
    )

    observation, next_offset = graph._format_sample_examples_observation(
        row_ids=[10],
        n=1,
        offset=0,
    )

    assert next_offset == 1
    assert "row_id=10" in observation
    assert "category=REFUND" in observation
    assert "intent=check_refund_status" in observation
    assert "customer_instruction=Where is my refund?" in observation
    assert "support_response=You can check your refund status in your account." in observation


def test_create_invocation_state_falls_back_to_initial_state_when_checkpoint_read_fails() -> None:
    class FailingCheckpointGraph:
        def get_state(self, config):
            raise RuntimeError("No checkpointer available in this test.")

    config = graph._build_graph_config(
        session_id="fallback_session",
        user_id="max",
        max_iterations=12,
    )

    result = graph._create_invocation_state(
        graph=FailingCheckpointGraph(),
        query="How many rows are in the dataset?",
        session_id="fallback_session",
        user_id="max",
        max_iterations=12,
        config=config,
    )

    assert result["session_id"] == "fallback_session"
    assert result["user_id"] == "max"
    assert result["route"] is None
    assert result["tool_trace"] == []
    assert result["last_structured_results"] == []
    assert isinstance(result["messages"][0], HumanMessage)
    assert result["messages"][0].content == "How many rows are in the dataset?"