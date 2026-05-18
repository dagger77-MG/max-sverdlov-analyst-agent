from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app import graph


@pytest.fixture(autouse=True)
def patch_checkpointer(monkeypatch: pytest.MonkeyPatch):
    """Keep graph unit tests independent from optional SQLite checkpoints."""
    graph.build_graph.cache_clear()

    if hasattr(graph.get_checkpointer, "cache_clear"):
        graph.get_checkpointer.cache_clear()

    monkeypatch.setattr(graph, "get_checkpointer", lambda: None)

    yield

    graph.build_graph.cache_clear()


class FakeActionLLM:
    def __init__(self, decisions: list[graph.AgentActionDecision]) -> None:
        self.decisions = decisions
        self.call_count = 0

    def invoke(self, messages):
        decision = self.decisions[self.call_count]
        self.call_count += 1
        return decision


class FakeProfileLLM:
    def __init__(self, observation: str = "") -> None:
        self.observation = observation

    def invoke(self, messages):
        return graph.ProfileObservationDecision(observation=self.observation)


def test_structured_query_uses_tools_and_returns_final_answer(monkeypatch) -> None:
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
        lambda user_id: type(
            "ProfileResult",
            (),
            {
                "user_id": user_id,
                "profile": "# User Profile\n\n- No durable user facts or preferences have been saved yet.\n",
            },
        )(),
    )
    monkeypatch.setattr(
        graph,
        "filter_rows_impl",
        lambda **kwargs: type(
            "FilterRowsResult",
            (),
            {
                "row_ids": [1, 2, 3],
                "match_count": 3,
                "applied_filters": kwargs,
            },
        )(),
    )
    monkeypatch.setattr(
        graph,
        "count_rows_impl",
        lambda row_ids=None: type(
            "CountRowsResult",
            (),
            {"count": len(row_ids or [])},
        )(),
    )
    monkeypatch.setattr(
        graph,
        "get_structured_profile_llm",
        lambda: FakeProfileLLM(),
    )

    fake_action_llm = FakeActionLLM(
        [
            graph.AgentActionDecision(
                thought="Find refund rows.",
                tool_name="filter_rows",
                tool_input={"category": "REFUND"},
            ),
            graph.AgentActionDecision(
                thought="Count the filtered rows.",
                tool_name="count_rows",
                tool_input={"row_ids": [1, 2, 3]},
            ),
            graph.AgentActionDecision(
                thought="Answer with the count.",
                tool_name="final_answer",
                final_answer="There are 3 refund rows in the dataset.",
            ),
        ]
    )
    monkeypatch.setattr(graph, "get_structured_action_llm", lambda: fake_action_llm)

    result = graph.invoke_agent(
        query="How many refund requests did we get?",
        session_id="test_session",
        user_id="max",
    )

    assert result["route"] == "structured"
    assert result["final_answer"] == "There are 3 refund rows in the dataset."
    assert [step["tool_name"] for step in result["tool_trace"]] == [
        "filter_rows",
        "count_rows",
    ]
    assert result["last_structured_results"][-1]["value"] == 3
    assert isinstance(result["messages"][-1], AIMessage)


def test_unstructured_query_uses_filter_and_summarize_tools(monkeypatch) -> None:
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
        lambda user_id: type(
            "ProfileResult",
            (),
            {"user_id": user_id, "profile": "# User Profile\n"},
        )(),
    )
    monkeypatch.setattr(
        graph,
        "filter_rows_impl",
        lambda **kwargs: type(
            "FilterRowsResult",
            (),
            {
                "row_ids": [10, 11],
                "match_count": 2,
                "applied_filters": kwargs,
            },
        )(),
    )
    monkeypatch.setattr(
        graph,
        "summarize_rows_impl",
        lambda row_ids, focus, max_examples=100: type(
            "SummarizeRowsResult",
            (),
            {
                "summary": "Rows reviewed: 2\nCustomers mainly provide product feedback.",
                "row_count_used": 2,
                "focus": focus,
            },
        )(),
    )
    monkeypatch.setattr(
        graph,
        "get_structured_profile_llm",
        lambda: FakeProfileLLM(),
    )

    fake_action_llm = FakeActionLLM(
        [
            graph.AgentActionDecision(
                thought="Find feedback rows.",
                tool_name="filter_rows",
                tool_input={"category": "FEEDBACK"},
            ),
            graph.AgentActionDecision(
                thought="Summarize the selected rows.",
                tool_name="summarize_rows",
                tool_input={
                    "row_ids": [10, 11],
                    "focus": "Summarize the FEEDBACK category.",
                    "max_examples": 100,
                },
            ),
            graph.AgentActionDecision(
                thought="Answer with the grounded summary.",
                tool_name="final_answer",
                final_answer="Customers mainly provide product feedback.",
            ),
        ]
    )
    monkeypatch.setattr(graph, "get_structured_action_llm", lambda: fake_action_llm)

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


def test_out_of_scope_query_refuses_without_data_tools(monkeypatch) -> None:
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
        lambda user_id: type(
            "ProfileResult",
            (),
            {"user_id": user_id, "profile": "# User Profile\n"},
        )(),
    )
    monkeypatch.setattr(
        graph,
        "get_structured_profile_llm",
        lambda: FakeProfileLLM(),
    )

    def fail_if_data_agent_called():
        raise AssertionError("Data agent should not be called for out-of-scope queries.")

    monkeypatch.setattr(graph, "get_structured_action_llm", fail_if_data_agent_called)

    result = graph.invoke_agent(
        query="Who won the World Cup?",
        session_id="test_session",
        user_id="max",
    )

    assert result["route"] == "out_of_scope"
    assert result["final_answer"] == graph.OUT_OF_SCOPE_REFUSAL
    assert result["tool_trace"] == []


def test_max_iteration_fallback(monkeypatch) -> None:
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
        lambda user_id: type(
            "ProfileResult",
            (),
            {"user_id": user_id, "profile": "# User Profile\n"},
        )(),
    )
    monkeypatch.setattr(
        graph,
        "get_structured_profile_llm",
        lambda: FakeProfileLLM(),
    )
    monkeypatch.setattr(
        graph,
        "get_dataset_schema_impl",
        lambda include_sample_values=True: type(
            "SchemaResult",
            (),
            {
                "row_count": 5,
                "columns": ["row_id", "instruction", "response", "category", "intent"],
            },
        )(),
    )

    repeated_decisions = [
        graph.AgentActionDecision(
            thought="Inspect schema again.",
            tool_name="get_dataset_schema",
            tool_input={"include_sample_values": False},
        )
        for _ in range(graph.settings.max_iterations)
    ]

    monkeypatch.setattr(
        graph,
        "get_structured_action_llm",
        lambda: FakeActionLLM(repeated_decisions),
    )

    result = graph.invoke_agent(
        query="Analyze the dataset.",
        session_id="test_session",
        user_id="max",
    )

    assert result["iteration_count"] == graph.settings.max_iterations
    assert result["final_answer"] == (
        "I could not complete the analysis within the allowed number of "
        "reasoning steps. Please try asking a more specific dataset question."
    )


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
        lambda user_id: type(
            "ProfileResult",
            (),
            {"user_id": user_id, "profile": "# User Profile\n"},
        )(),
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


def test_route_specific_instructions_for_structured_route() -> None:
    instructions = graph._route_specific_instructions("structured")

    assert "STRUCTURED queries" in instructions
    assert "Prefer deterministic tools" in instructions
    assert "Do not use summarize_rows unless" in instructions


def test_route_specific_instructions_for_unstructured_route() -> None:
    instructions = graph._route_specific_instructions("unstructured")

    assert "UNSTRUCTURED queries" in instructions
    assert "Then use summarize_rows" in instructions
    assert "Do not answer from general knowledge" in instructions


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
    assert result["iteration_count"] == 0
    assert result["final_answer"] is None
    assert "last_structured_results" not in result
    assert isinstance(result["messages"][0], HumanMessage)
    assert result["messages"][0].content == "Show me 3 more."


def test_load_user_profile_node_returns_partial_state_update(monkeypatch) -> None:
    monkeypatch.setattr(
        graph,
        "read_user_profile_impl",
        lambda user_id: type(
            "ProfileResult",
            (),
            {"user_id": user_id, "profile": "# User Profile\n\n- Test profile\n"},
        )(),
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
    state["tool_trace"] = [
        {
            "tool_name": "sample_examples",
            "tool_input": {
                "row_ids": [10, 11, 12, 13, 14, 15],
                "n": 3,
                "offset": 0,
            },
            "observation": "Returned 3 examples. Next offset = 3.",
        }
    ]
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
                                "instruction": "Example 4",
                            },
                        )(),
                        type(
                            "ExampleRow",
                            (),
                            {
                                "row_id": 14,
                                "instruction": "Example 5",
                            },
                        )(),
                        type(
                            "ExampleRow",
                            (),
                            {
                                "row_id": 15,
                                "instruction": "Example 6",
                            },
                        )(),
                    ],
                    "next_offset": 6,
                },
            )()
        ),
    )

    fake_action_llm = FakeActionLLM(
        [
            graph.AgentActionDecision(
                thought="Continue from the previous sample offset.",
                tool_name="sample_examples",
                tool_input={
                    "row_ids": [10, 11, 12, 13, 14, 15],
                    "n": 3,
                    "offset": 3,
                },
            ),
            graph.AgentActionDecision(
                thought="Answer with the next examples.",
                tool_name="final_answer",
                final_answer="Here are the next 3 examples: 13, 14, and 15.",
            ),
        ]
    )
    monkeypatch.setattr(graph, "get_structured_action_llm", lambda: fake_action_llm)

    result = graph.react_data_agent_node(state)

    assert captured_inputs == [
        {
            "row_ids": [10, 11, 12, 13, 14, 15],
            "n": 3,
            "offset": 3,
        }
    ]
    assert result["final_answer"] == "Here are the next 3 examples: 13, 14, and 15."
    assert state["last_structured_results"][-1] == {
        "label": "sample_examples",
        "value": 6,
        "query_type": "sample",
        "row_ids": [10, 11, 12, 13, 14, 15],
    }


def test_follow_up_what_about_refunds_preserves_count_pattern(monkeypatch) -> None:
    state = graph.create_initial_state(
        query="What about refunds?",
        session_id="test_session",
        user_id="max",
    )
    state["route"] = "structured"
    state["route_reason"] = "The user asks a follow-up structured count question."
    state["user_profile"] = "# User Profile\n"
    state["last_structured_results"] = [
        {
            "label": "count_rows",
            "value": 5,
            "query_type": "count",
            "row_ids": [1, 2, 3, 4, 5],
        }
    ]

    monkeypatch.setattr(
        graph,
        "filter_rows_impl",
        lambda **kwargs: type(
            "FilterRowsResult",
            (),
            {
                "row_ids": [10, 11, 12],
                "match_count": 3,
                "applied_filters": kwargs,
            },
        )(),
    )
    monkeypatch.setattr(
        graph,
        "count_rows_impl",
        lambda row_ids=None: type(
            "CountRowsResult",
            (),
            {"count": len(row_ids or [])},
        )(),
    )

    fake_action_llm = FakeActionLLM(
        [
            graph.AgentActionDecision(
                thought="Find refund rows like the previous count question.",
                tool_name="filter_rows",
                tool_input={"category": "REFUND"},
            ),
            graph.AgentActionDecision(
                thought="Count the refund rows.",
                tool_name="count_rows",
                tool_input={"row_ids": [10, 11, 12]},
            ),
            graph.AgentActionDecision(
                thought="Answer with the refund count.",
                tool_name="final_answer",
                final_answer="There are 3 refund rows.",
            ),
        ]
    )
    monkeypatch.setattr(graph, "get_structured_action_llm", lambda: fake_action_llm)

    result = graph.react_data_agent_node(state)

    assert result["final_answer"] == "There are 3 refund rows."
    assert [step["tool_name"] for step in result["tool_trace"]] == [
        "filter_rows",
        "count_rows",
    ]
    assert state["last_structured_results"][-1]["value"] == 3


def test_follow_up_total_count_of_last_two_uses_stored_structured_results(
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
            "query_type": "count",
            "row_ids": None,
        },
    ]

    def fail_if_tool_called(*args, **kwargs):
        raise AssertionError("No dataset tool should be needed for stored count totals.")

    monkeypatch.setattr(graph, "filter_rows_impl", fail_if_tool_called)
    monkeypatch.setattr(graph, "count_rows_impl", fail_if_tool_called)
    monkeypatch.setattr(graph, "sample_examples_impl", fail_if_tool_called)
    monkeypatch.setattr(graph, "group_counts_impl", fail_if_tool_called)
    monkeypatch.setattr(graph, "summarize_rows_impl", fail_if_tool_called)

    fake_action_llm = FakeActionLLM(
        [
            graph.AgentActionDecision(
                thought="Use the last two stored count results: 514 and 842.",
                tool_name="final_answer",
                final_answer="The total count of the last two results is 1,356.",
            ),
        ]
    )
    monkeypatch.setattr(graph, "get_structured_action_llm", lambda: fake_action_llm)

    result = graph.react_data_agent_node(state)

    assert result["final_answer"] == "The total count of the last two results is 1,356."
    assert result["tool_trace"] == []