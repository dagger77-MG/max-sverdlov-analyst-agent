from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage

from app.agent import tool_executor


def _state(query: str = "How many refund requests?") -> dict[str, Any]:
    return {
        "messages": [HumanMessage(content=query)],
        "session_id": "test_session",
        "user_id": "max",
        "route": "structured",
        "route_reason": "test route reason",
        "tool_trace": [],
        "last_structured_results": [],
        "user_profile": "# User Profile\n",
        "max_iterations": 12,
        "final_answer": None,
    }


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


def test_normalize_resolve_filter_value_input_accepts_aliases() -> None:
    result = tool_executor._normalize_resolve_filter_value_input(
        {
            "filter_value": "refund requests",
            "columns": ["category"],
            "top_k": 3,
        }
    )

    assert result == {
        "query": "refund requests",
        "columns": ["category"],
        "top_k": 3,
    }


def test_normalize_resolve_filter_value_input_defaults_columns_and_top_k() -> None:
    result = tool_executor._normalize_resolve_filter_value_input(
        {
            "value": "shipping",
        }
    )

    assert result == {
        "query": "shipping",
        "columns": ["category", "intent"],
        "top_k": 5,
    }


def test_canonical_tool_input_for_count_rows_keeps_only_filters() -> None:
    result = tool_executor._canonical_tool_input(
        "count_rows",
        {
            "category": "REFUND",
            "intent": None,
            "text_query": None,
            "unused": "ignored",
        },
    )

    assert result == {
        "category": "REFUND",
        "intent": None,
        "text_query": None,
    }


def test_canonical_tool_input_for_sample_examples_defaults_n_and_offset() -> None:
    result = tool_executor._canonical_tool_input(
        "sample_examples",
        {
            "category": "REFUND",
        },
    )

    assert result == {
        "category": "REFUND",
        "intent": None,
        "text_query": None,
        "n": 3,
        "offset": 0,
    }


def test_tool_call_already_exists_uses_canonical_inputs() -> None:
    state = _state()
    state["tool_trace"] = [
        {
            "tool_name": "sample_examples",
            "tool_input": {
                "category": "REFUND",
                "intent": None,
                "text_query": None,
                "n": 3,
                "offset": 0,
            },
            "observation": "Returned 3 examples.",
        }
    ]

    assert tool_executor._tool_call_already_exists(
        state,
        "sample_examples",
        {
            "category": "REFUND",
        },
    )


def test_format_sample_examples_observation_includes_full_example_details(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        tool_executor,
        "sample_examples_impl",
        lambda category=None, intent=None, text_query=None, n=3, offset=0: _model_result(
            examples=[
                _example_row(
                    row_id=10,
                    category="REFUND",
                    intent="check_refund_status",
                    instruction="Where is my refund?",
                    response="You can check your refund status in your account.",
                )
            ],
            next_offset=1,
            match_count=1,
            applied_filters={
                "category": category,
                "intent": intent,
                "text_query": text_query,
            },
        ),
    )

    observation, next_offset, match_count, applied_filters = (
        tool_executor._format_sample_examples_observation(
            filters={
                "category": "REFUND",
                "intent": None,
                "text_query": None,
            },
            n=1,
            offset=0,
        )
    )

    assert next_offset == 1
    assert match_count == 1
    assert applied_filters == {
        "category": "REFUND",
        "intent": None,
        "text_query": None,
    }
    assert "row_id=10" in observation
    assert "category=REFUND" in observation
    assert "intent=check_refund_status" in observation
    assert "customer_instruction=Where is my refund?" in observation
    assert "support_response=You can check your refund status in your account." in observation


def test_execute_resolve_filter_value_appends_trace_and_structured_result(
    monkeypatch,
) -> None:
    state = _state("How many refund requests?")

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

    tool_executor._execute_selected_tool(
        state,
        "resolve_filter_value",
        {
            "query": "refund requests",
            "columns": ["category", "intent"],
            "top_k": 5,
        },
    )

    assert state["tool_trace"][-1]["tool_name"] == "resolve_filter_value"
    assert state["tool_trace"][-1]["tool_input"] == {
        "query": "refund requests",
        "columns": ["category", "intent"],
        "top_k": 5,
    }

    observation = json.loads(state["tool_trace"][-1]["observation"])
    assert observation["confidence"] == "high"
    assert observation["recommended_filter"] == {
        "category": "REFUND",
        "intent": None,
    }

    assert state["last_structured_results"][-1] == {
        "label": "resolve_filter_value:refund requests",
        "value": "high",
        "query_type": "resolve_filter_value",
        "filters": {
            "category": "REFUND",
            "intent": None,
        },
        "match_count": None,
    }


def test_execute_count_rows_appends_trace_and_structured_result(monkeypatch) -> None:
    state = _state("How many refund requests?")

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

    tool_executor._execute_selected_tool(
        state,
        "count_rows",
        {
            "category": "REFUND",
        },
    )

    assert state["tool_trace"][-1]["tool_name"] == "count_rows"
    assert state["tool_trace"][-1]["tool_input"] == {
        "category": "REFUND",
        "intent": None,
        "text_query": None,
    }
    assert json.loads(state["tool_trace"][-1]["observation"]) == {
        "count": 2992,
        "applied_filters": {
            "category": "REFUND",
            "intent": None,
            "text_query": None,
        },
    }
    assert state["last_structured_results"][-1] == {
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


def test_execute_sample_examples_appends_trace_and_structured_result(monkeypatch) -> None:
    state = _state("Show me 1 example from the REFUND category.")

    monkeypatch.setattr(
        tool_executor,
        "sample_examples_impl",
        lambda category=None, intent=None, text_query=None, n=3, offset=0: _model_result(
            examples=[
                _example_row(
                    row_id=10,
                    category="REFUND",
                    intent="check_refund_status",
                    instruction="Where is my refund?",
                    response="You can check your refund status in your account.",
                )
            ],
            next_offset=1,
            match_count=6,
            applied_filters={
                "category": category,
                "intent": intent,
                "text_query": text_query,
            },
        ),
    )

    tool_executor._execute_selected_tool(
        state,
        "sample_examples",
        {
            "category": "REFUND",
            "n": 1,
            "offset": 0,
        },
    )

    assert state["tool_trace"][-1]["tool_name"] == "sample_examples"
    assert state["tool_trace"][-1]["tool_input"] == {
        "category": "REFUND",
        "intent": None,
        "text_query": None,
        "n": 1,
        "offset": 0,
    }
    assert "row_id=10" in state["tool_trace"][-1]["observation"]
    assert state["last_structured_results"][-1] == {
        "label": "sample_examples",
        "value": 1,
        "query_type": "sample",
        "filters": {
            "category": "REFUND",
            "intent": None,
            "text_query": None,
        },
        "match_count": 6,
    }


def test_group_counts_rejects_missing_category_filter_for_scoped_intent_distribution(
    monkeypatch,
) -> None:
    state = _state("What is the distribution of intents in the ACCOUNT category?")

    def fail_if_group_counts_called(group_by, category=None, intent=None, text_query=None, top_k=20):
        raise AssertionError(
            "group_counts_impl should not be called for an unfiltered scoped distribution."
        )

    monkeypatch.setattr(tool_executor, "group_counts_impl", fail_if_group_counts_called)

    tool_executor._execute_selected_tool(
        state=state,
        tool_name="group_counts",
        tool_input={
            "group_by": "intent",
            "top_k": 5,
        },
    )

    assert len(state["tool_trace"]) == 1
    assert state["tool_trace"][0]["tool_name"] == "group_counts"
    assert state["last_structured_results"] == []

    observation = json.loads(state["tool_trace"][0]["observation"])
    assert observation == {
        "error": (
            "group_counts needs a category filter for an intent "
            "distribution inside a category."
        ),
        "required_next_step": (
            "Resolve the category value, then call group_counts with "
            "group_by='intent' and category=<resolved_category>."
        ),
    }


def test_group_counts_expands_top_k_for_full_distribution_without_explicit_top_k(
    monkeypatch,
) -> None:
    state = _state("What is the distribution of intents?")

    captured_inputs: list[dict[str, Any]] = []

    def fake_group_counts_impl(
        group_by,
        category=None,
        intent=None,
        text_query=None,
        top_k=20,
    ):
        captured_inputs.append(
            {
                "group_by": group_by,
                "category": category,
                "intent": intent,
                "text_query": text_query,
                "top_k": top_k,
            }
        )
        return _model_result(
            group_by=group_by,
            counts=[
                {
                    "label": "recover_password",
                    "count": 997,
                }
            ],
            match_count=5986,
            applied_filters={
                "category": category,
                "intent": intent,
                "text_query": text_query,
            },
        )

    monkeypatch.setattr(tool_executor, "group_counts_impl", fake_group_counts_impl)

    tool_executor._execute_selected_tool(
        state=state,
        tool_name="group_counts",
        tool_input={
            "group_by": "intent",
            "top_k": 5,
        },
    )

    assert captured_inputs == [
        {
            "group_by": "intent",
            "category": None,
            "intent": None,
            "text_query": None,
            "top_k": 20,
        }
    ]
    assert state["tool_trace"][-1]["tool_input"]["top_k"] == 20
    assert state["last_structured_results"][-1] == {
        "label": "group_counts:intent",
        "value": 1,
        "query_type": "group_counts",
        "filters": {
            "category": None,
            "intent": None,
            "text_query": None,
        },
        "match_count": 5986,
    }


def test_execute_summarize_rows_uses_latest_user_message_as_default_focus(
    monkeypatch,
) -> None:
    state = _state("How do agents respond to refund requests?")
    captured_inputs: list[dict[str, Any]] = []

    def fake_summarize_rows_impl(
        category=None,
        intent=None,
        text_query=None,
        focus="",
        target_field="both",
        max_examples=100,
    ):
        captured_inputs.append(
            {
                "category": category,
                "intent": intent,
                "text_query": text_query,
                "focus": focus,
                "target_field": target_field,
                "max_examples": max_examples,
            }
        )
        return _model_result(
            summary="Agents explain refund steps.",
            row_count_used=100,
            match_count=2992,
            focus=focus,
            target_field=target_field,
            applied_filters={
                "category": category,
                "intent": intent,
                "text_query": text_query,
            },
        )

    monkeypatch.setattr(
        tool_executor,
        "summarize_rows_impl",
        fake_summarize_rows_impl,
    )

    tool_executor._execute_selected_tool(
        state=state,
        tool_name="summarize_rows",
        tool_input={
            "category": "REFUND",
            "target_field": "response",
            "max_examples": 100,
        },
    )

    assert captured_inputs == [
        {
            "category": "REFUND",
            "intent": None,
            "text_query": None,
            "focus": "How do agents respond to refund requests?",
            "target_field": "response",
            "max_examples": 100,
        }
    ]
    assert state["last_structured_results"][-1] == {
        "label": "summarize_rows",
        "value": 100,
        "query_type": "summary",
        "filters": {
            "category": "REFUND",
            "intent": None,
            "text_query": None,
        },
        "match_count": 2992,
    }


def test_execute_read_user_profile_appends_profile_observation(monkeypatch) -> None:
    state = _state("What do you remember about me?")

    monkeypatch.setattr(
        tool_executor,
        "read_user_profile_impl",
        lambda user_id: _model_result(
            user_id=user_id,
            profile="# User Profile\n\n- User prefers concise answers.\n",
        ),
    )

    tool_executor._execute_selected_tool(
        state=state,
        tool_name="read_user_profile",
        tool_input={},
    )

    assert state["tool_trace"][-1]["tool_name"] == "read_user_profile"
    assert state["tool_trace"][-1]["tool_input"] == {
        "user_id": "max",
    }
    assert json.loads(state["tool_trace"][-1]["observation"]) == {
        "user_id": "max",
        "profile": "# User Profile\n\n- User prefers concise answers.\n",
    }


def test_unknown_tool_raises_value_error() -> None:
    state = _state("How many rows are there?")

    try:
        tool_executor._execute_selected_tool(
            state=state,
            tool_name="unknown_tool",
            tool_input={},
        )
    except ValueError as exc:
        assert "Unknown tool selected by planner: unknown_tool" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unknown tool.")