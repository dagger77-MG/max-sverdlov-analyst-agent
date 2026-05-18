from __future__ import annotations

import pandas as pd
import pytest

from app import tools


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_id": [0, 1, 2, 3, 4],
            "instruction": [
                "I want a refund for my order",
                "How can I contact customer service?",
                "I want to give feedback about the product",
                "My refund has not arrived",
                "I have a billing complaint",
            ],
            "response": [
                "You can request a refund through your account.",
                "You can contact our support team by email.",
                "Thank you for sharing your feedback.",
                "Refunds usually take several business days.",
                "Please share the billing issue details.",
            ],
            "category": [
                "REFUND",
                "CONTACT",
                "FEEDBACK",
                "REFUND",
                "COMPLAINT",
            ],
            "intent": [
                "get_refund",
                "contact_customer_service",
                "give_feedback",
                "check_refund_status",
                "billing_complaint",
            ],
        }
    )


@pytest.fixture(autouse=True)
def patch_dataset(monkeypatch: pytest.MonkeyPatch, sample_df: pd.DataFrame) -> None:
    monkeypatch.setattr(tools, "get_dataset_df", lambda: sample_df)


def test_get_dataset_schema_returns_columns_and_row_count(
    monkeypatch: pytest.MonkeyPatch,
    sample_df: pd.DataFrame,
) -> None:
    monkeypatch.setattr(
        tools,
        "get_dataset_metadata",
        lambda include_sample_values=True: {
            "columns": list(sample_df.columns),
            "row_count": len(sample_df),
            "sample_values": {
                "category": ["REFUND", "CONTACT", "FEEDBACK"],
            }
            if include_sample_values
            else None,
        },
    )

    result = tools.get_dataset_schema_impl(include_sample_values=True)

    assert result.columns == list(sample_df.columns)
    assert result.row_count == 5
    assert result.sample_values == {
        "category": ["REFUND", "CONTACT", "FEEDBACK"],
    }


def test_count_rows_counts_full_dataset() -> None:
    result = tools.count_rows_impl()

    assert result.count == 5


def test_count_rows_counts_selected_row_ids() -> None:
    result = tools.count_rows_impl(row_ids=[0, 2, 4])

    assert result.count == 3


def test_filter_rows_filters_by_category_case_insensitive() -> None:
    result = tools.filter_rows_impl(category="refund")

    assert result.match_count == 2
    assert result.row_ids == [0, 3]


def test_filter_rows_filters_by_intent_case_insensitive() -> None:
    result = tools.filter_rows_impl(intent="GET_REFUND")

    assert result.match_count == 1
    assert result.row_ids == [0]


def test_filter_rows_filters_by_text_query() -> None:
    result = tools.filter_rows_impl(text_query="billing")

    assert result.match_count == 1
    assert result.row_ids == [4]


def test_filter_rows_applies_limit_but_keeps_total_match_count() -> None:
    result = tools.filter_rows_impl(category="REFUND", limit=1)

    assert result.match_count == 2
    assert result.row_ids == [0]


def test_sample_examples_returns_requested_number_of_examples() -> None:
    result = tools.sample_examples_impl(row_ids=[0, 1, 2], n=2)

    assert len(result.examples) == 2
    assert result.examples[0].row_id == 0
    assert result.examples[1].row_id == 1
    assert result.next_offset == 2


def test_sample_examples_respects_offset() -> None:
    result = tools.sample_examples_impl(row_ids=[0, 1, 2, 3], n=2, offset=2)

    assert len(result.examples) == 2
    assert result.examples[0].row_id == 2
    assert result.examples[1].row_id == 3
    assert result.next_offset == 4


def test_group_counts_groups_by_category_sorted_descending() -> None:
    result = tools.group_counts_impl(group_by="category")

    assert result.group_by == "category"
    assert result.counts[0].label == "REFUND"
    assert result.counts[0].count == 2


def test_group_counts_groups_selected_row_ids_only() -> None:
    result = tools.group_counts_impl(group_by="category", row_ids=[0, 3])

    assert len(result.counts) == 1
    assert result.counts[0].label == "REFUND"
    assert result.counts[0].count == 2


def test_rows_to_summary_context_formats_selected_rows(sample_df: pd.DataFrame) -> None:
    context = tools._rows_to_summary_context(sample_df.head(1))

    assert "row_id=0" in context
    assert "category=REFUND" in context
    assert "intent=get_refund" in context
    assert "customer_instruction=I want a refund for my order" in context
    assert "support_response=You can request a refund through your account." in context


def test_summarize_rows_falls_back_to_deterministic_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        tools,
        "get_summarizer_llm",
        lambda: (_ for _ in ()).throw(RuntimeError("No summarizer available.")),
    )

    result = tools.summarize_rows_impl(
        row_ids=[0, 3],
        focus="refund requests",
        max_examples=10,
    )

    assert result.row_count_used == 2
    assert "Rows reviewed: 2" in result.summary
    assert "Top categories: REFUND (2)" in result.summary


def test_summarize_rows_returns_non_empty_summary() -> None:
    result = tools.summarize_rows_impl(
        row_ids=[0, 3],
        focus="refund requests",
        max_examples=10,
    )

    assert result.row_count_used == 2
    assert result.focus == "refund requests"
    assert "Rows reviewed: 2" in result.summary
    assert "REFUND" in result.summary
    assert "refund" in result.summary.lower()


def test_summarize_rows_uses_llm_summary_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeLLMResponse:
        content = "Refund rows mainly describe customers asking for refunds or refund status."

    class FakeSummarizerLLM:
        def __init__(self) -> None:
            self.received_messages = None

        def invoke(self, messages):
            self.received_messages = messages
            return FakeLLMResponse()

    fake_llm = FakeSummarizerLLM()
    monkeypatch.setattr(tools, "get_summarizer_llm", lambda: fake_llm)

    result = tools.summarize_rows_impl(
        row_ids=[0, 3],
        focus="refund requests",
        max_examples=10,
    )

    assert result.summary == (
        "Refund rows mainly describe customers asking for refunds or refund status."
    )
    assert result.row_count_used == 2
    assert fake_llm.received_messages is not None
    assert "Focus: refund requests" in fake_llm.received_messages[1].content

    
def test_schema_based_wrappers_call_implementations() -> None:
    filter_result = tools.filter_rows(
        tools.FilterRowsInput(category="REFUND"),
    )
    count_result = tools.count_rows(
        tools.CountRowsInput(row_ids=filter_result.row_ids),
    )
    examples_result = tools.sample_examples(
        tools.SampleExamplesInput(row_ids=filter_result.row_ids, n=1),
    )
    group_result = tools.group_counts(
        tools.GroupCountsInput(group_by="category"),
    )
    summary_result = tools.summarize_rows(
        tools.SummarizeRowsInput(row_ids=filter_result.row_ids, focus="refund"),
    )

    assert filter_result.match_count == 2
    assert count_result.count == 2
    assert len(examples_result.examples) == 1
    assert group_result.counts[0].label == "REFUND"
    assert summary_result.row_count_used == 2