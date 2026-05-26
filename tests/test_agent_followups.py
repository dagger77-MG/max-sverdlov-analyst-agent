from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage

from app.agent import followups


def _state(query: str = "Show me 3 more.") -> dict[str, Any]:
    return {
        "messages": [HumanMessage(content=query)],
        "session_id": "test_session",
        "user_id": "max",
        "route": "structured",
        "route_reason": "The user asks for more examples.",
        "tool_trace": [],
        "last_structured_results": [],
        "user_profile": "# User Profile\n",
        "max_iterations": 12,
        "final_answer": None,
    }


def test_is_more_examples_query_detects_common_follow_up_phrases() -> None:
    assert followups._is_more_examples_query("Show me 3 more.")
    assert followups._is_more_examples_query("Give me another 5 examples")
    assert followups._is_more_examples_query("next 2 rows")
    assert followups._is_more_examples_query("additional samples")


def test_is_more_examples_query_rejects_non_example_follow_ups() -> None:
    assert not followups._is_more_examples_query("What is the total of the last two?")
    assert not followups._is_more_examples_query("Summarize the previous result.")
    assert not followups._is_more_examples_query("")


def test_requested_example_count_extracts_requested_number() -> None:
    assert followups._requested_example_count("Show me 7 more.") == 7
    assert followups._requested_example_count("Give me another 4 examples.") == 4
    assert followups._requested_example_count("next 2 rows") == 2


def test_requested_example_count_defaults_and_clamps() -> None:
    assert followups._requested_example_count("Show me more examples.") == 3
    assert followups._requested_example_count("Show me 999 more.") == 20
    assert followups._requested_example_count("Show me 0 more.") == 1


def test_latest_sample_context_returns_most_recent_sample_result() -> None:
    result = followups._latest_sample_context(
        [
            {
                "label": "count_rows",
                "value": 10,
                "query_type": "count",
                "filters": {
                    "category": "REFUND",
                    "intent": None,
                    "text_query": None,
                },
                "match_count": 10,
            },
            {
                "label": "sample_examples",
                "value": 6,
                "query_type": "sample",
                "filters": {
                    "category": "SHIPPING",
                    "intent": None,
                    "text_query": None,
                },
                "match_count": 20,
            },
        ]
    )

    assert result == (
        {
            "category": "SHIPPING",
            "intent": None,
            "text_query": None,
        },
        6,
    )


def test_latest_sample_context_returns_none_without_sample_result() -> None:
    result = followups._latest_sample_context(
        [
            {
                "label": "count_rows",
                "value": 10,
                "query_type": "count",
                "filters": {
                    "category": "REFUND",
                    "intent": None,
                    "text_query": None,
                },
                "match_count": 10,
            }
        ]
    )

    assert result is None


def test_handle_more_examples_follow_up_returns_none_without_more_examples_query() -> None:
    state = _state("What is the total of the last two?")
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

    result = followups._handle_more_examples_follow_up(state)

    assert result is None
    assert state["tool_trace"] == []


def test_handle_more_examples_follow_up_returns_none_without_prior_sample_context() -> None:
    state = _state("Show me 3 more.")
    state["last_structured_results"] = []

    result = followups._handle_more_examples_follow_up(state)

    assert result is None
    assert state["tool_trace"] == []


def test_handle_more_examples_follow_up_appends_trace_and_structured_result() -> None:
    state = _state("Show me 2 more.")
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

    captured_formatter_inputs: list[tuple[dict[str, str | None], int, int]] = []

    def fake_sample_formatter(
        filters: dict[str, str | None],
        n: int,
        offset: int,
    ) -> tuple[str, int, int, dict[str, str | None]]:
        captured_formatter_inputs.append((filters, n, offset))
        return (
            "Returned 2 examples from 6 matching rows. Next offset = 5.",
            5,
            6,
            {
                "category": "REFUND",
                "intent": None,
                "text_query": None,
            },
        )

    result = followups._handle_more_examples_follow_up(
        state,
        sample_formatter=fake_sample_formatter,
    )

    assert captured_formatter_inputs == [
        (
            {
                "category": "REFUND",
                "intent": None,
                "text_query": None,
            },
            2,
            3,
        )
    ]
    assert result is not None
    assert result["final_answer"] == (
        "Returned 2 examples from 6 matching rows. Next offset = 5."
    )
    assert state["final_answer"] == (
        "Returned 2 examples from 6 matching rows. Next offset = 5."
    )
    assert state["tool_trace"][0] == {
        "event_type": "tool",
        "tool_name": "sample_examples",
        "tool_input": {
            "category": "REFUND",
            "intent": None,
            "text_query": None,
            "n": 2,
            "offset": 3,
        },
        "observation": "Returned 2 examples from 6 matching rows. Next offset = 5.",
    }
    assert state["tool_trace"][1] == {
        "event_type": "reviewer",
        "reviewer_status": "answered",
        "reviewer_reason": (
            "Planner and reviewer LLMs skipped: user asked for more examples "
            "from the previous sample context, so deterministic follow-up "
            "pagination produced the final answer."
        ),
        "reviewer_final_answer": "",
        "suggested_tool_name": "",
        "suggested_tool_input": {},
    }
    assert state["last_structured_results"][-1] == {
        "label": "sample_examples",
        "value": 5,
        "query_type": "sample",
        "filters": {
            "category": "REFUND",
            "intent": None,
            "text_query": None,
        },
        "match_count": 6,
    }