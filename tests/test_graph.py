from __future__ import annotations

from langchain_core.messages import AIMessage

from app import graph


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