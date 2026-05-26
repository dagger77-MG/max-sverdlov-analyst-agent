from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from app.agent import evidence_contracts


def _state(query: str) -> dict[str, Any]:
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


def test_final_answer_update_sets_state_and_returns_partial_update() -> None:
    state = _state("How many refund requests?")

    result = evidence_contracts._final_answer_update(
        state,
        "There are 2,992 refund requests.",
    )

    assert state["final_answer"] == "There are 2,992 refund requests."
    assert result["final_answer"] == "There are 2,992 refund requests."
    assert isinstance(result["messages"][0], AIMessage)
    assert result["messages"][0].content == "There are 2,992 refund requests."


def test_trace_observation_is_error_detects_error_payload() -> None:
    assert evidence_contracts._trace_observation_is_error(
        json.dumps(
            {
                "error": "group_counts needs a category filter.",
                "required_next_step": "Resolve category first.",
            }
        )
    )


def test_trace_observation_is_error_rejects_non_error_payloads() -> None:
    assert not evidence_contracts._trace_observation_is_error(
        json.dumps({"count": 2992})
    )
    assert not evidence_contracts._trace_observation_is_error("not json")


def test_failed_explicit_intent_resolver_returns_cannot_answer_update() -> None:
    state = _state("Show me 3 examples from the SHIPPING intent.")
    state["tool_trace"] = [
        {
            "tool_name": "resolve_filter_value",
            "tool_input": {
                "query": "SHIPPING",
                "columns": ["intent"],
                "top_k": 5,
            },
            "observation": json.dumps(
                {
                    "query": "SHIPPING",
                    "candidates": [],
                    "recommended_filter": {
                        "category": None,
                        "intent": None,
                    },
                    "confidence": "none",
                }
            ),
        }
    ]

    result = evidence_contracts._return_failed_explicit_resolver_answer_if_ready(
        state
    )

    assert result is not None
    assert result["final_answer"] == (
        'No matching intent value exists for "SHIPPING" in the dataset, '
        "so I can't answer that request using that intent."
    )
    assert state["final_answer"] == result["final_answer"]


def test_failed_explicit_category_resolver_returns_cannot_answer_update() -> None:
    state = _state("Show me 3 examples from the UNKNOWN category.")
    state["tool_trace"] = [
        {
            "tool_name": "resolve_filter_value",
            "tool_input": {
                "query": "UNKNOWN",
                "columns": ["category"],
                "top_k": 5,
            },
            "observation": json.dumps(
                {
                    "query": "UNKNOWN",
                    "candidates": [],
                    "recommended_filter": {
                        "category": None,
                        "intent": None,
                    },
                    "confidence": "none",
                }
            ),
        }
    ]

    result = evidence_contracts._return_failed_explicit_resolver_answer_if_ready(
        state
    )

    assert result is not None
    assert result["final_answer"] == (
        'No matching category value exists for "UNKNOWN" in the dataset, '
        "so I can't answer that request using that category."
    )


def test_failed_resolver_helper_ignores_ambiguous_resolver_miss() -> None:
    state = _state("Show me examples about shipping.")
    state["tool_trace"] = [
        {
            "tool_name": "resolve_filter_value",
            "tool_input": {
                "query": "shipping",
                "columns": ["category", "intent"],
                "top_k": 5,
            },
            "observation": json.dumps(
                {
                    "query": "shipping",
                    "candidates": [],
                    "recommended_filter": {
                        "category": None,
                        "intent": None,
                    },
                    "confidence": "none",
                }
            ),
        }
    ]

    assert (
        evidence_contracts._return_failed_explicit_resolver_answer_if_ready(state)
        is None
    )


def test_sample_examples_observation_has_examples() -> None:
    assert evidence_contracts._sample_examples_observation_has_examples(
        "Returned 2 examples from 6 matching rows. Next offset = 5."
    )


def test_sample_examples_observation_without_examples_is_not_valid_shortcut() -> None:
    assert not evidence_contracts._sample_examples_observation_has_examples(
        "Returned 0 examples from 0 matching rows. Next offset = 0."
    )


def test_deterministic_sample_examples_answer_requires_actual_examples() -> None:
    state = _state("Show me 2 examples from the REFUND category.")
    state["tool_trace"] = [
        {
            "tool_name": "sample_examples",
            "tool_input": {
                "category": "REFUND",
                "intent": None,
                "text_query": None,
                "n": 2,
                "offset": 0,
            },
            "observation": (
                "Returned 2 examples from 6 matching rows. Next offset = 2.\n\n"
                "row_id=10\n"
                "category=REFUND\n"
                "intent=get_refund\n"
                "customer_instruction=Where is my refund?\n"
                "support_response=You can check your refund status in your account."
            ),
        }
    ]

    result = evidence_contracts._return_deterministic_sample_examples_answer_if_ready(
        state=state,
        tool_name="sample_examples",
    )

    assert result is not None
    assert result["final_answer"].startswith("Returned 2 examples from 6 matching rows.")
    assert state["final_answer"] == result["final_answer"]
    assert state["tool_trace"][-1] == {
        "event_type": "reviewer",
        "reviewer_status": "answered",
        "reviewer_reason": (
            "Reviewer LLM skipped: sample_examples returned requested row "
            "content, so deterministic fast path produced the final answer."
        ),
        "reviewer_final_answer": "",
        "suggested_tool_name": "",
        "suggested_tool_input": {},
    }
    assert result["tool_trace"][-1] == state["tool_trace"][-1]


def test_deterministic_sample_examples_answer_ignores_zero_example_observation() -> None:
    state = _state("Show me 2 examples from the REFUND category.")
    state["tool_trace"] = [
        {
            "tool_name": "sample_examples",
            "tool_input": {
                "category": "REFUND",
                "intent": None,
                "text_query": None,
                "n": 2,
                "offset": 0,
            },
            "observation": "Returned 0 examples from 0 matching rows. Next offset = 0.",
        }
    ]

    result = evidence_contracts._return_deterministic_sample_examples_answer_if_ready(
        state=state,
        tool_name="sample_examples",
    )

    assert result is None
    assert state["final_answer"] is None


def test_answer_contract_blocks_global_category_distribution_without_group_counts() -> None:
    state = _state("Break down the dataset by category.")

    error = evidence_contracts._answer_contract_error(state)

    assert error == (
        "The user asked for a full-dataset category distribution. "
        "Valid evidence requires an unfiltered group_counts call with "
        "group_by='category'."
    )


def test_answer_contract_allows_global_category_distribution_with_unfiltered_group_counts() -> None:
    state = _state("Break down the dataset by category.")
    state["tool_trace"] = [
        {
            "tool_name": "group_counts",
            "tool_input": {
                "group_by": "category",
                "category": None,
                "intent": None,
                "text_query": None,
                "top_k": 20,
            },
            "observation": json.dumps(
                {
                    "group_by": "category",
                    "counts": [
                        {
                            "label": "ACCOUNT",
                            "count": 5986,
                        }
                    ],
                    "match_count": 26872,
                }
            ),
        }
    ]

    assert evidence_contracts._answer_contract_error(state) is None


def test_answer_contract_blocks_global_category_distribution_with_filtered_group_counts() -> None:
    state = _state("Break down the dataset by category.")
    state["tool_trace"] = [
        {
            "tool_name": "group_counts",
            "tool_input": {
                "group_by": "category",
                "category": "REFUND",
                "intent": None,
                "text_query": None,
                "top_k": 20,
            },
            "observation": json.dumps(
                {
                    "group_by": "category",
                    "counts": [
                        {
                            "label": "REFUND",
                            "count": 2992,
                        }
                    ],
                    "match_count": 2992,
                }
            ),
        }
    ]

    error = evidence_contracts._answer_contract_error(state)

    assert error == (
        "The user asked for a full-dataset category distribution. "
        "Valid evidence requires an unfiltered group_counts call with "
        "group_by='category'."
    )


def test_answer_contract_blocks_scoped_intent_distribution_without_resolver() -> None:
    state = _state("What is the distribution of intents in the ACCOUNT category?")

    error = evidence_contracts._answer_contract_error(state)

    assert error is not None
    assert "intent distribution inside a category" in error
    assert "resolving the category" in error
    assert "group_counts" in error


def test_answer_contract_blocks_scoped_intent_distribution_after_resolver_only() -> None:
    state = _state("What is the distribution of intents in the ACCOUNT category?")
    state["tool_trace"] = [
        {
            "tool_name": "resolve_filter_value",
            "tool_input": {
                "query": "ACCOUNT",
                "columns": ["category"],
                "top_k": 5,
            },
            "observation": json.dumps(
                {
                    "query": "ACCOUNT",
                    "recommended_filter": {
                        "category": "ACCOUNT",
                        "intent": None,
                    },
                    "confidence": "high",
                }
            ),
        }
    ]

    error = evidence_contracts._answer_contract_error(state)

    assert error == (
        "The category has been resolved, but group_counts has not been "
        "called with group_by='intent' and that category filter yet."
    )


def test_answer_contract_allows_scoped_intent_distribution_with_filtered_group_counts() -> None:
    state = _state("What is the distribution of intents in the ACCOUNT category?")
    state["tool_trace"] = [
        {
            "tool_name": "group_counts",
            "tool_input": {
                "group_by": "intent",
                "category": "ACCOUNT",
                "intent": None,
                "text_query": None,
                "top_k": 20,
            },
            "observation": json.dumps(
                {
                    "group_by": "intent",
                    "counts": [
                        {
                            "label": "recover_password",
                            "count": 997,
                        }
                    ],
                    "match_count": 5986,
                }
            ),
        }
    ]

    assert evidence_contracts._answer_contract_error(state) is None


def test_answer_contract_blocks_scoped_category_distribution_without_resolver() -> None:
    state = _state("Break down track_refund by category.")

    error = evidence_contracts._answer_contract_error(state)

    assert error is not None
    assert "category distribution inside an intent" in error
    assert "resolving the intent" in error
    assert "group_counts" in error


def test_answer_contract_blocks_scoped_category_distribution_after_resolver_only() -> None:
    state = _state("Break down track_refund by category.")
    state["tool_trace"] = [
        {
            "tool_name": "resolve_filter_value",
            "tool_input": {
                "query": "track_refund",
                "columns": ["intent"],
                "top_k": 5,
            },
            "observation": json.dumps(
                {
                    "query": "track_refund",
                    "recommended_filter": {
                        "category": None,
                        "intent": "track_refund",
                    },
                    "confidence": "high",
                }
            ),
        }
    ]

    error = evidence_contracts._answer_contract_error(state)

    assert error == (
        "The intent has been resolved, but group_counts has not been "
        "called with group_by='category' and that intent filter yet."
    )


def test_answer_contract_allows_scoped_category_distribution_with_filtered_group_counts() -> None:
    state = _state("Break down track_refund by category.")
    state["tool_trace"] = [
        {
            "tool_name": "group_counts",
            "tool_input": {
                "group_by": "category",
                "category": None,
                "intent": "track_refund",
                "text_query": None,
                "top_k": 20,
            },
            "observation": json.dumps(
                {
                    "group_by": "category",
                    "counts": [
                        {
                            "label": "REFUND",
                            "count": 11,
                        }
                    ],
                    "match_count": 11,
                }
            ),
        }
    ]

    assert evidence_contracts._answer_contract_error(state) is None


def test_answer_contract_rejects_error_group_counts_observation() -> None:
    state = _state("Break down the dataset by category.")
    state["tool_trace"] = [
        {
            "tool_name": "group_counts",
            "tool_input": {
                "group_by": "category",
                "category": None,
                "intent": None,
                "text_query": None,
                "top_k": 20,
            },
            "observation": json.dumps(
                {
                    "error": "group_counts failed.",
                    "required_next_step": "Retry with valid input.",
                }
            ),
        }
    ]

    error = evidence_contracts._answer_contract_error(state)

    assert error == (
        "The user asked for a full-dataset category distribution. "
        "Valid evidence requires an unfiltered group_counts call with "
        "group_by='category'."
    )


def test_semantic_summary_contract_blocks_text_query_summary_after_resolution() -> None:
    state = _state(
        "How do customer service representatives typically respond to cancellation requests?"
    )
    state["tool_trace"] = [
        {
            "tool_name": "summarize_rows",
            "tool_input": {
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
            "observation": json.dumps(
                {
                    "summary": "Weak text-query-only summary.",
                    "row_count_used": 1,
                    "match_count": 1,
                    "applied_filters": {
                        "category": None,
                        "intent": None,
                        "text_query": "cancellation requests",
                    },
                }
            ),
        },
        {
            "tool_name": "resolve_filter_value",
            "tool_input": {
                "query": "cancellation requests",
                "columns": ["category", "intent"],
                "top_k": 5,
            },
            "observation": json.dumps(
                {
                    "query": "cancellation requests",
                    "recommended_filter": {
                        "category": "CANCEL",
                        "intent": None,
                    },
                    "confidence": "high",
                }
            ),
        },
    ]

    error = evidence_contracts._semantic_summary_contract_error(state)

    assert error is not None
    assert 'category="CANCEL"' in error
    assert "text_query-only summarize_rows call is not valid final evidence" in error


def test_semantic_summary_contract_allows_later_semantic_summary() -> None:
    state = _state(
        "How do customer service representatives typically respond to cancellation requests?"
    )
    state["tool_trace"] = [
        {
            "tool_name": "summarize_rows",
            "tool_input": {
                "category": None,
                "intent": None,
                "text_query": "cancellation requests",
                "focus": "response patterns",
                "target_field": "response",
                "max_examples": 100,
            },
            "observation": json.dumps(
                {
                    "summary": "Weak text-query-only summary.",
                    "row_count_used": 1,
                    "match_count": 1,
                }
            ),
        },
        {
            "tool_name": "resolve_filter_value",
            "tool_input": {
                "query": "cancellation requests",
                "columns": ["category", "intent"],
                "top_k": 5,
            },
            "observation": json.dumps(
                {
                    "query": "cancellation requests",
                    "recommended_filter": {
                        "category": "CANCEL",
                        "intent": None,
                    },
                    "confidence": "high",
                }
            ),
        },
        {
            "tool_name": "summarize_rows",
            "tool_input": {
                "category": "CANCEL",
                "intent": None,
                "text_query": None,
                "focus": "response patterns",
                "target_field": "response",
                "max_examples": 100,
            },
            "observation": json.dumps(
                {
                    "summary": "Agents explain cancellation steps.",
                    "row_count_used": 100,
                    "match_count": 950,
                }
            ),
        },
    ]

    assert evidence_contracts._semantic_summary_contract_error(state) is None