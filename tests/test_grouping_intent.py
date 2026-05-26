from __future__ import annotations

import pytest

from app.agent.grouping_intent import analyze_grouping_request


@pytest.mark.parametrize(
    "query",
    [
        "Break down the dataset by category",
        "Break down all by category",
        "Show the distribution of categories",
        "Category distribution",
        "Count by category",
    ],
)
def test_global_category_grouping(query: str) -> None:
    result = analyze_grouping_request(query)

    assert result.is_grouping_request is True
    assert result.requested_group_by == "category"
    assert result.is_scoped is False
    assert result.required_filter_column is None
    assert result.scope_phrase is None


@pytest.mark.parametrize(
    "query",
    [
        "Break down the dataset by intent",
        "Break down all by intent",
        "Show the distribution of intents",
       "Intent distribution",
        "Count by intent",
    ],
)
def test_global_intent_grouping(query: str) -> None:
    result = analyze_grouping_request(query)

    assert result.is_grouping_request is True
    assert result.requested_group_by == "intent"
    assert result.is_scoped is False
    assert result.required_filter_column is None
    assert result.scope_phrase is None


@pytest.mark.parametrize(
    "query",
    [
        "Break down ACCOUNT by intent",
        "Intent breakdown for ACCOUNT",
        "What intents appear under ACCOUNT?",
        "Which intents occur inside ACCOUNT?",
        "Show intents within ACCOUNT",
    ],
)
def test_scoped_category_to_intent_grouping(query: str) -> None:
    result = analyze_grouping_request(query)

    assert result.is_grouping_request is True
    assert result.requested_group_by == "intent"
    assert result.is_scoped is True
    assert result.required_filter_column == "category"
    assert result.scope_phrase == "ACCOUNT"


@pytest.mark.parametrize(
    ("query", "expected_scope"),
    [
        ("Break down track_refund by category", "track_refund"),
        ("Category breakdown for track_refund", "track_refund"),
        ("What categories appear under track_refund?", "track_refund"),
        ("Which categories occur inside check_refund_status?", "check_refund_status"),
        ("Show categories within delete_account", "delete_account"),
    ],
)
def test_scoped_intent_to_category_grouping(
    query: str,
    expected_scope: str,
) -> None:
    result = analyze_grouping_request(query)

    assert result.is_grouping_request is True
    assert result.requested_group_by == "category"
    assert result.is_scoped is True
    assert result.required_filter_column == "intent"
    assert result.scope_phrase == expected_scope


@pytest.mark.parametrize(
    "query",
    [
        "How many refund requests did we get?",
        "Show me 5 examples of the SHIPPING category.",
        "Summarize the FEEDBACK category.",
    ],
)
def test_non_grouping_controls(query: str) -> None:
    result = analyze_grouping_request(query)

    assert result.is_grouping_request is False
    assert result.requested_group_by is None
    assert result.is_scoped is False
    assert result.required_filter_column is None
    assert result.scope_phrase is None