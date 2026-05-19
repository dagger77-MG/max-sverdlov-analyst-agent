from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app import graph


@pytest.fixture(autouse=True)
def patch_checkpointer(monkeypatch: pytest.MonkeyPatch):
    """Keep graph unit tests independent from optional SQLite checkpoints."""
    graph.build_graph.cache_clear()

    if hasattr(graph.get_checkpointer, "cache_clear"):
        graph.get_checkpointer.cache_clear()

    if hasattr(graph.get_langchain_data_agent, "cache_clear"):
        graph.get_langchain_data_agent.cache_clear()

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


class FakeLangChainAgent:
    def __init__(self, result_messages):
        self.result_messages = result_messages
        self.received_input = None
        self.received_config = None

    def invoke(self, input_data, config=None):
        self.received_input = input_data
        self.received_config = config
        return {
            "messages": self.result_messages,
        }


class FailingLangChainAgent:
    def invoke(self, input_data, config=None):
        raise RuntimeError("Simulated agent failure.")


def _profile_result(user_id: str, profile: str = "# User Profile\n"):
    return type(
        "ProfileResult",
        (),
        {
            "user_id": user_id,
            "profile": profile,
        },
    )()


def test_structured_query_uses_standard_langchain_agent_and_returns_trace(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        graph,
        "route_query_with_reason",
        lambda query: graph.RouteDecision(
            route="structured",
            reason="The user asks for an exact refund count.",
        ),
    )
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

    fake_agent = FakeLangChainAgent(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_filter",
                        "name": "filter_rows",
                        "args": {"category": "REFUND"},
                    }
                ],
            ),
            ToolMessage(
                content=str(
                    {
                        "row_ids": [1, 2, 3],
                        "match_count": 3,
                        "applied_filters": {
                            "category": "REFUND",
                            "intent": None,
                            "text_query": None,
                            "limit": None,
                        },
                    }
                ),
                name="filter_rows",
                tool_call_id="call_filter",
            ),
            AIMessage(content="There are 3 refund rows in the dataset."),
        ]
    )
    monkeypatch.setattr(graph, "get_langchain_data_agent", lambda: fake_agent)

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
    assert fake_agent.received_config == {"recursion_limit": result["max_iterations"] + 5}


def test_assignment_question_categories_exist_uses_group_counts(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        graph,
        "route_query_with_reason",
        lambda query: graph.RouteDecision(
            route="structured",
            reason="The user asks which categories exist in the dataset.",
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
        lambda: FakeProfileLLM(),
    )

    fake_agent = FakeLangChainAgent(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_group",
                        "name": "group_counts",
                        "args": {"group_by": "category", "top_k": 100},
                    }
                ],
            ),
            ToolMessage(
                content=str(
                    {
                        "group_by": "category",
                        "counts": [
                            {"label": "REFUND", "count": 842},
                            {"label": "SHIPPING", "count": 420},
                            {"label": "ACCOUNT", "count": 300},
                        ],
                    }
                ),
                name="group_counts",
                tool_call_id="call_group",
            ),
            AIMessage(content="The dataset categories include REFUND, SHIPPING, and ACCOUNT."),
        ]
    )
    monkeypatch.setattr(graph, "get_langchain_data_agent", lambda: fake_agent)

    result = graph.invoke_agent(
        query="What categories exist in the dataset?",
        session_id="test_session",
        user_id="max",
    )

    assert result["route"] == "structured"
    assert result["final_answer"] == (
        "The dataset categories include REFUND, SHIPPING, and ACCOUNT."
    )
    assert [step["tool_name"] for step in result["tool_trace"]] == ["group_counts"]
    assert result["tool_trace"][0]["tool_input"] == {
        "group_by": "category",
        "top_k": 100,
    }


def test_assignment_question_shipping_examples_uses_filter_then_sample(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        graph,
        "route_query_with_reason",
        lambda query: graph.RouteDecision(
            route="structured",
            reason="The user asks for examples from a dataset category.",
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
        lambda: FakeProfileLLM(),
    )

    fake_agent = FakeLangChainAgent(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_filter",
                        "name": "filter_rows",
                        "args": {"category": "SHIPPING"},
                    }
                ],
            ),
            ToolMessage(
                content=str(
                    {
                        "row_ids": [20, 21, 22, 23, 24, 25],
                        "match_count": 6,
                        "applied_filters": {
                            "category": "SHIPPING",
                            "intent": None,
                            "text_query": None,
                            "limit": None,
                        },
                    }
                ),
                name="filter_rows",
                tool_call_id="call_filter",
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_sample",
                        "name": "sample_examples",
                        "args": {
                            "row_ids": [20, 21, 22, 23, 24, 25],
                            "n": 5,
                            "offset": 0,
                        },
                    }
                ],
            ),
            ToolMessage(
                content=str(
                    {
                        "examples": [
                            {
                                "row_id": 20,
                                "instruction": "Where is my package?",
                                "response": "You can track it from your account.",
                                "category": "SHIPPING",
                                "intent": "track_order",
                            }
                        ],
                        "next_offset": 5,
                    }
                ),
                name="sample_examples",
                tool_call_id="call_sample",
            ),
            AIMessage(content="Here are 5 SHIPPING examples from the dataset."),
        ]
    )
    monkeypatch.setattr(graph, "get_langchain_data_agent", lambda: fake_agent)

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
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        graph,
        "route_query_with_reason",
        lambda query: graph.RouteDecision(
            route="structured",
            reason="The user asks for intent distribution within a category.",
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
        lambda: FakeProfileLLM(),
    )

    fake_agent = FakeLangChainAgent(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_filter",
                        "name": "filter_rows",
                        "args": {"category": "ACCOUNT"},
                    }
                ],
            ),
            ToolMessage(
                content=str(
                    {
                        "row_ids": [30, 31, 32],
                        "match_count": 3,
                        "applied_filters": {
                            "category": "ACCOUNT",
                            "intent": None,
                            "text_query": None,
                            "limit": None,
                        },
                    }
                ),
                name="filter_rows",
                tool_call_id="call_filter",
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_group",
                        "name": "group_counts",
                        "args": {
                            "group_by": "intent",
                            "row_ids": [30, 31, 32],
                            "top_k": 20,
                        },
                    }
                ],
            ),
            ToolMessage(
                content=str(
                    {
                        "group_by": "intent",
                        "counts": [
                            {"label": "recover_password", "count": 2},
                            {"label": "delete_account", "count": 1},
                        ],
                    }
                ),
                name="group_counts",
                tool_call_id="call_group",
            ),
            AIMessage(content="In ACCOUNT, recover_password appears 2 times and delete_account appears once."),
        ]
    )
    monkeypatch.setattr(graph, "get_langchain_data_agent", lambda: fake_agent)

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


def test_assignment_question_money_back_alias_uses_refund_filter_then_sample(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        graph,
        "route_query_with_reason",
        lambda query: graph.RouteDecision(
            route="structured",
            reason="The user asks for refund-related examples using a natural-language alias.",
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
        lambda: FakeProfileLLM(),
    )

    fake_agent = FakeLangChainAgent(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_filter",
                        "name": "filter_rows",
                        "args": {"category": "money back"},
                    }
                ],
            ),
            ToolMessage(
                content=str(
                    {
                        "row_ids": [40, 41, 42],
                        "match_count": 3,
                        "applied_filters": {
                            "category": "money back",
                            "intent": None,
                            "text_query": None,
                            "limit": None,
                        },
                    }
                ),
                name="filter_rows",
                tool_call_id="call_filter",
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_sample",
                        "name": "sample_examples",
                        "args": {
                            "row_ids": [40, 41, 42],
                            "n": 3,
                            "offset": 0,
                        },
                    }
                ],
            ),
            ToolMessage(
                content=str(
                    {
                        "examples": [
                            {
                                "row_id": 40,
                                "instruction": "I want my money back.",
                                "response": "You can request a refund through your account.",
                                "category": "REFUND",
                                "intent": "get_refund",
                            }
                        ],
                        "next_offset": 3,
                    }
                ),
                name="sample_examples",
                tool_call_id="call_sample",
            ),
            AIMessage(content="Here are refund-related examples for people wanting their money back."),
        ]
    )
    monkeypatch.setattr(graph, "get_langchain_data_agent", lambda: fake_agent)

    result = graph.invoke_agent(
        query="Show me examples of people wanting their money back.",
        session_id="test_session",
        user_id="max",
    )

    assert result["route"] == "structured"
    assert [step["tool_name"] for step in result["tool_trace"]] == [
        "filter_rows",
        "sample_examples",
    ]
    assert result["tool_trace"][0]["tool_input"] == {"category": "money back"}
    assert result["tool_trace"][1]["tool_input"] == {
        "row_ids": [40, 41, 42],
        "n": 3,
        "offset": 0,
    }


def test_unstructured_query_uses_standard_langchain_agent_trace(monkeypatch) -> None:
    monkeypatch.setattr(
        graph,
        "route_query_with_reason",
        lambda query: graph.RouteDecision(
            route="unstructured",
            reason="The user asks for a qualitative category summary.",
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
        lambda: FakeProfileLLM(),
    )

    fake_agent = FakeLangChainAgent(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_filter",
                        "name": "filter_rows",
                        "args": {"category": "FEEDBACK"},
                    }
                ],
            ),
            ToolMessage(
                content=str(
                    {
                        "row_ids": [10, 11],
                        "match_count": 2,
                        "applied_filters": {"category": "FEEDBACK"},
                    }
                ),
                name="filter_rows",
                tool_call_id="call_filter",
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_summary",
                        "name": "summarize_rows",
                        "args": {
                            "row_ids": [10, 11],
                            "focus": "Summarize the FEEDBACK category.",
                            "max_examples": 100,
                        },
                    }
                ],
            ),
            ToolMessage(
                content=str(
                    {
                        "summary": "Customers mainly provide product feedback.",
                        "row_count_used": 2,
                        "focus": "Summarize the FEEDBACK category.",
                    }
                ),
                name="summarize_rows",
                tool_call_id="call_summary",
            ),
            AIMessage(content="Customers mainly provide product feedback."),
        ]
    )
    monkeypatch.setattr(graph, "get_langchain_data_agent", lambda: fake_agent)

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


def test_assignment_question_cancellation_response_summary_uses_selected_rows(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        graph,
        "route_query_with_reason",
        lambda query: graph.RouteDecision(
            route="unstructured",
            reason="The user asks for qualitative analysis of cancellation requests.",
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
        lambda: FakeProfileLLM(),
    )

    fake_agent = FakeLangChainAgent(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_filter",
                        "name": "filter_rows",
                        "args": {"text_query": "cancellation"},
                    }
                ],
            ),
            ToolMessage(
                content=str(
                    {
                        "row_ids": [50, 51],
                        "match_count": 2,
                        "applied_filters": {
                            "category": None,
                            "intent": None,
                            "text_query": "cancellation",
                            "limit": None,
                        },
                    }
                ),
                name="filter_rows",
                tool_call_id="call_filter",
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_summary",
                        "name": "summarize_rows",
                        "args": {
                            "row_ids": [50, 51],
                            "focus": (
                                "How do customer service representatives typically "
                                "respond to cancellation requests?"
                            ),
                            "max_examples": 100,
                        },
                    }
                ],
            ),
            ToolMessage(
                content=str(
                    {
                        "summary": "Agents usually acknowledge the cancellation request and explain the required account steps.",
                        "row_count_used": 2,
                        "focus": "cancellation response patterns",
                    }
                ),
                name="summarize_rows",
                tool_call_id="call_summary",
            ),
            AIMessage(
                content=(
                    "Agents usually acknowledge the cancellation request and explain "
                    "the required account steps."
                )
            ),
        ]
    )
    monkeypatch.setattr(graph, "get_langchain_data_agent", lambda: fake_agent)

    result = graph.invoke_agent(
        query="How do customer service representatives typically respond to cancellation requests?",
        session_id="test_session",
        user_id="max",
    )

    assert result["route"] == "unstructured"
    assert [step["tool_name"] for step in result["tool_trace"]] == [
        "filter_rows",
        "summarize_rows",
    ]
    assert result["tool_trace"][0]["tool_input"] == {"text_query": "cancellation"}
    assert result["tool_trace"][1]["tool_input"]["row_ids"] == [50, 51]
    assert "cancellation request" in result["final_answer"]


def test_assignment_question_complaint_intent_response_summary_uses_selected_rows(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        graph,
        "route_query_with_reason",
        lambda query: graph.RouteDecision(
            route="unstructured",
            reason="The user asks for qualitative analysis of complaint response patterns.",
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
        lambda: FakeProfileLLM(),
    )

    fake_agent = FakeLangChainAgent(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_filter",
                        "name": "filter_rows",
                        "args": {"category": "complaint"},
                    }
                ],
            ),
            ToolMessage(
                content=str(
                    {
                        "row_ids": [60, 61],
                        "match_count": 2,
                        "applied_filters": {
                            "category": "complaint",
                            "intent": None,
                            "text_query": None,
                            "limit": None,
                        },
                    }
                ),
                name="filter_rows",
                tool_call_id="call_filter",
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_summary",
                        "name": "summarize_rows",
                        "args": {
                            "row_ids": [60, 61],
                            "focus": "Summarize how agents respond to complaint intents.",
                            "max_examples": 100,
                        },
                    }
                ],
            ),
            ToolMessage(
                content=str(
                    {
                        "summary": (
                            "Agents acknowledge the complaint, ask for issue details, "
                            "and guide the customer toward the next support step."
                        ),
                        "row_count_used": 2,
                        "focus": "complaint response patterns",
                    }
                ),
                name="summarize_rows",
                tool_call_id="call_summary",
            ),
            AIMessage(
                content=(
                    "Agents acknowledge the complaint, ask for issue details, "
                    "and guide the customer toward the next support step."
                )
            ),
        ]
    )
    monkeypatch.setattr(graph, "get_langchain_data_agent", lambda: fake_agent)

    result = graph.invoke_agent(
        query="Summarize how agents respond to complaint intents.",
        session_id="test_session",
        user_id="max",
    )

    assert result["route"] == "unstructured"
    assert [step["tool_name"] for step in result["tool_trace"]] == [
        "filter_rows",
        "summarize_rows",
    ]
    assert result["tool_trace"][0]["tool_input"] == {"category": "complaint"}
    assert result["tool_trace"][1]["tool_input"] == {
        "row_ids": [60, 61],
        "focus": "Summarize how agents respond to complaint intents.",
        "max_examples": 100,
    }
    assert "acknowledge the complaint" in result["final_answer"]


def test_out_of_scope_query_refuses_without_data_agent(monkeypatch) -> None:
    monkeypatch.setattr(
        graph,
        "route_query_with_reason",
        lambda query: graph.RouteDecision(
            route="out_of_scope",
            reason="The user asks about external facts.",
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
        lambda: FakeProfileLLM(),
    )

    def fail_if_data_agent_called():
        raise AssertionError("Data agent should not be called for out-of-scope queries.")

    monkeypatch.setattr(graph, "get_langchain_data_agent", fail_if_data_agent_called)

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
def test_assignment_out_of_scope_questions_refuse_without_data_agent(
    monkeypatch,
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
    monkeypatch.setattr(
        graph,
        "read_user_profile_impl",
        lambda user_id: _profile_result(user_id),
    )
    monkeypatch.setattr(
        graph,
        "get_structured_profile_llm",
        lambda: FakeProfileLLM(),
    )

    def fail_if_data_agent_called():
        raise AssertionError("Data agent should not be called for out-of-scope queries.")

    monkeypatch.setattr(graph, "get_langchain_data_agent", fail_if_data_agent_called)

    result = graph.invoke_agent(
        query=query,
        session_id="test_session",
        user_id="max",
    )

    assert result["route"] == "out_of_scope"
    assert result["final_answer"] == graph.OUT_OF_SCOPE_REFUSAL
    assert result["tool_trace"] == []


def test_standard_agent_failure_returns_graceful_fallback(monkeypatch) -> None:
    monkeypatch.setattr(
        graph,
        "route_query_with_reason",
        lambda query: graph.RouteDecision(
            route="structured",
            reason="The user asks a dataset question.",
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
        lambda: FakeProfileLLM(),
    )
    monkeypatch.setattr(
        graph,
        "get_langchain_data_agent",
        lambda: FailingLangChainAgent(),
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


def test_profile_update_node_saves_durable_observation(monkeypatch) -> None:
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


def test_load_user_profile_node_returns_partial_state_update(monkeypatch) -> None:
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


def test_router_node_returns_partial_state_update(monkeypatch) -> None:
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


def test_route_after_router_sends_dataset_queries_to_langchain_agent() -> None:
    state = graph.create_initial_state(
        query="How many refund requests?",
        session_id="test_session",
        user_id="max",
    )
    state["route"] = "structured"

    assert graph.route_after_router(state) == "langchain_data_agent_node"

    state["route"] = "unstructured"

    assert graph.route_after_router(state) == "langchain_data_agent_node"


def test_route_after_router_sends_out_of_scope_to_refusal_node() -> None:
    state = graph.create_initial_state(
        query="Who won the World Cup?",
        session_id="test_session",
        user_id="max",
    )
    state["route"] = "out_of_scope"

    assert graph.route_after_router(state) == "refusal_node"


def test_follow_up_show_me_three_more_uses_previous_row_ids_and_offset(
    monkeypatch,
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

    def fail_if_agent_called():
        raise AssertionError("The standard agent should not be called for deterministic follow-up examples.")

    monkeypatch.setattr(graph, "get_langchain_data_agent", fail_if_agent_called)

    result = graph.langchain_data_agent_node(state)

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


def test_total_of_last_two_follow_up_goes_to_standard_agent_with_context(
    monkeypatch,
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

    fake_agent = FakeLangChainAgent(
        [
            AIMessage(content="The total count of the last two results is 1,356."),
        ]
    )

    monkeypatch.setattr(graph, "get_langchain_data_agent", lambda: fake_agent)
    result = graph.langchain_data_agent_node(state)

    context_message = fake_agent.received_input["messages"][0].content

    assert "complaints" in context_message
    assert "514" in context_message
    assert "refunds" in context_message
    assert "842" in context_message
    assert result["final_answer"] == "The total count of the last two results is 1,356."


def test_sample_examples_trace_includes_full_example_details(monkeypatch) -> None:
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


def test_extract_langchain_tool_trace_and_results_for_count_rows() -> None:
    messages = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "call_count",
                    "name": "count_rows",
                    "args": {"row_ids": [1, 2, 3]},
                }
            ],
        ),
        ToolMessage(
            content=str({"count": 3}),
            name="count_rows",
            tool_call_id="call_count",
        ),
        AIMessage(content="There are 3 rows."),
    ]

    trace, structured_results = graph._extract_langchain_tool_trace_and_results(messages)

    assert trace == [
        {
            "tool_name": "count_rows",
            "tool_input": {"row_ids": [1, 2, 3]},
            "observation": "{'count': 3}",
        }
    ]
    assert structured_results == [
        {
            "label": "count_rows",
            "value": 3,
            "query_type": "count",
            "row_ids": [1, 2, 3],
        }
    ]


def test_extract_langchain_tool_trace_and_results_for_sample_examples() -> None:
    messages = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "call_sample",
                    "name": "sample_examples",
                    "args": {
                        "row_ids": [10, 11, 12],
                        "n": 2,
                        "offset": 0,
                    },
                }
            ],
        ),
        ToolMessage(
            content=str(
                {
                    "examples": [
                        {
                            "row_id": 10,
                            "instruction": "Example 1",
                            "response": "Response 1",
                            "category": "REFUND",
                            "intent": "get_refund",
                        }
                    ],
                    "next_offset": 1,
                }
            ),
            name="sample_examples",
            tool_call_id="call_sample",
        ),
        AIMessage(content="Here is one example."),
    ]

    trace, structured_results = graph._extract_langchain_tool_trace_and_results(messages)

    assert trace[0]["tool_name"] == "sample_examples"
    assert structured_results == [
        {
            "label": "sample_examples",
            "value": 1,
            "query_type": "sample",
            "row_ids": [10, 11, 12],
        }
    ]


def test_extract_final_answer_skips_ai_messages_with_tool_calls() -> None:
    messages = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "call_filter",
                    "name": "filter_rows",
                    "args": {"category": "REFUND"},
                }
            ],
        ),
        ToolMessage(
            content=str({"row_ids": [1], "match_count": 1}),
            name="filter_rows",
            tool_call_id="call_filter",
        ),
        AIMessage(content="There is 1 refund row."),
    ]

    assert graph._extract_final_answer(messages) == "There is 1 refund row."


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