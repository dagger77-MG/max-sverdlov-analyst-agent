from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.agent import context


def _state(
    query: str = "How many refund requests?",
    user_profile: str = "# User Profile\n\n- User prefers concise answers.\n",
) -> dict:
    return {
        "messages": [HumanMessage(content=query)],
        "session_id": "test_session",
        "user_id": "max",
        "route": "structured",
        "route_reason": "The user asks for an exact dataset operation.",
        "tool_trace": [],
        "last_structured_results": [],
        "user_profile": user_profile,
        "max_iterations": 12,
        "final_answer": None,
    }


def test_latest_user_message_returns_latest_human_message() -> None:
    messages = [
        HumanMessage(content="First question"),
        AIMessage(content="First answer"),
        HumanMessage(content="Second question"),
    ]

    assert context.latest_user_message(messages) == "Second question"


def test_latest_user_message_returns_empty_string_without_human_message() -> None:
    assert context.latest_user_message([AIMessage(content="Only assistant")]) == ""


def test_structured_results_for_prompt_returns_default_when_empty() -> None:
    assert (
        context._structured_results_for_prompt([])
        == "No recent structured results."
    )


def test_structured_results_for_prompt_includes_filters_not_row_ids() -> None:
    prompt_text = context._structured_results_for_prompt(
        [
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
            }
        ]
    )

    assert "label=refunds" in prompt_text
    assert "value=842" in prompt_text
    assert "query_type=count" in prompt_text
    assert '"category": "REFUND"' in prompt_text
    assert "row_ids" not in prompt_text


def test_structured_results_for_prompt_limits_to_last_five_results() -> None:
    results = [
        {
            "label": f"result_{index}",
            "value": index,
            "query_type": "count",
            "filters": {
                "category": f"CATEGORY_{index}",
                "intent": None,
                "text_query": None,
            },
            "match_count": index,
        }
        for index in range(7)
    ]

    prompt_text = context._structured_results_for_prompt(results)

    assert "result_0" not in prompt_text
    assert "result_1" not in prompt_text
    assert "result_2" in prompt_text
    assert "result_6" in prompt_text


def test_compact_tool_input_for_prompt_returns_shallow_copy() -> None:
    original = {
        "category": "REFUND",
        "n": 3,
    }

    compacted = context._compact_tool_input_for_prompt(original)

    assert compacted == original
    assert compacted is not original


def test_compact_tool_trace_for_prompt_returns_default_when_empty() -> None:
    assert (
        context._compact_tool_trace_for_prompt([])
        == "No tool calls yet in this turn."
    )


def test_compact_tool_trace_for_prompt_formats_tool_steps() -> None:
    prompt_text = context._compact_tool_trace_for_prompt(
        [
            {
                "tool_name": "count_rows",
                "tool_input": {
                    "category": "REFUND",
                    "intent": None,
                    "text_query": None,
                },
                "observation": '{"count": 842}',
            }
        ]
    )

    assert "1. tool=count_rows" in prompt_text
    assert '"category": "REFUND"' in prompt_text
    assert 'observation={"count": 842}' in prompt_text


def test_profile_context_hidden_for_dataset_planning_by_default() -> None:
    state = _state(query="How many refund requests?")

    profile_context = context._profile_context_for_planner(state)

    assert "Profile hidden for dataset tool planning." in profile_context
    assert "User prefers concise answers" not in profile_context


def test_profile_context_visible_for_explicit_profile_question() -> None:
    state = _state(query="What do you remember about me? Show my profile.")

    profile_context = context._profile_context_for_planner(state)

    assert "User prefers concise answers." in profile_context


def test_build_planner_messages_includes_route_results_trace_and_feedback() -> None:
    state = _state()
    state["last_structured_results"] = [
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
        }
    ]
    state["tool_trace"] = [
        {
            "tool_name": "count_rows",
            "tool_input": {
                "category": "REFUND",
                "intent": None,
                "text_query": None,
            },
            "observation": '{"count": 842}',
        }
    ]

    messages = context._build_planner_messages(
        state,
        reviewer_feedback="Need one more tool.",
    )

    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], SystemMessage)
    assert isinstance(messages[2], HumanMessage)
    assert messages[2].content == "How many refund requests?"

    context_message = messages[1].content
    assert "Current route: structured" in context_message
    assert "Route reason: The user asks for an exact dataset operation." in context_message
    assert "label=refunds" in context_message
    assert "tool=count_rows" in context_message
    assert "Need one more tool." in context_message


def test_build_reviewer_messages_includes_trace_without_profile() -> None:
    state = _state()
    state["tool_trace"] = [
        {
            "tool_name": "count_rows",
            "tool_input": {
                "category": "REFUND",
                "intent": None,
                "text_query": None,
            },
            "observation": '{"count": 842}',
        }
    ]

    messages = context._build_reviewer_messages(state)

    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], SystemMessage)
    assert isinstance(messages[2], HumanMessage)
    assert messages[2].content == "How many refund requests?"

    reviewer_context = messages[1].content
    assert "Current route: structured" in reviewer_context
    assert "tool=count_rows" in reviewer_context
    assert "User profile:" not in reviewer_context