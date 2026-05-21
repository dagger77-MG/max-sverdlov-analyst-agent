from __future__ import annotations

import pandas as pd
import pytest

from app import tools


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_id": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "instruction": [
                "I want a refund for my order",
                "How can I contact customer service?",
                "I want to give feedback about the product",
                "My refund has not arrived",
                "I have a billing complaint",
                "Where is my package?",
                "My shipment is late",
                "I forgot my account password",
                "I want to delete my account",
                "I want to cancel my subscription",
            ],
            "response": [
                "You can request a refund through your account.",
                "You can contact our support team by email.",
                "Thank you for sharing your feedback.",
                "Refunds usually take several business days.",
                "Please share the billing issue details.",
                "You can track your package from your account.",
                "Please check the shipping status page for delivery updates.",
                "You can recover your password from the account login page.",
                "You can delete your account from account settings.",
                "You can start the cancellation process from your subscription settings.",
            ],
            "category": [
                "REFUND",
                "CONTACT",
                "FEEDBACK",
                "REFUND",
                "COMPLAINT",
                "SHIPPING",
                "SHIPPING",
                "ACCOUNT",
                "ACCOUNT",
                "CANCELLATION",
            ],
            "intent": [
                "get_refund",
                "contact_customer_service",
                "give_feedback",
                "check_refund_status",
                "billing_complaint",
                "track_order",
                "check_shipping_status",
                "recover_password",
                "delete_account",
                "cancel_subscription",
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
    assert result.row_count == len(sample_df)
    assert result.sample_values == {
        "category": ["REFUND", "CONTACT", "FEEDBACK"],
    }


def test_count_rows_counts_full_dataset(sample_df: pd.DataFrame) -> None:
    result = tools.count_rows_impl()

    assert result.count == len(sample_df)


def test_count_rows_counts_selected_row_ids() -> None:
    result = tools.count_rows_impl(row_ids=[0, 2, 4])

    assert result.count == 3


def test_filter_rows_filters_by_category_case_insensitive() -> None:
    result = tools.filter_rows_impl(category="refund")

    assert result.match_count == 2
    assert result.row_ids == [0, 3]


@pytest.mark.parametrize(
    "category_alias",
    [
        "refunds",
        "refund requests",
        "reimbursement",
        "reimbursement cases",
        "money back",
        "guarantee",
    ],
)
def test_filter_rows_normalizes_refund_category_aliases(
    category_alias: str,
) -> None:
    result = tools.filter_rows_impl(category=category_alias)

    assert result.match_count == 2
    assert result.row_ids == [0, 3]


def test_assignment_money_back_alias_maps_to_refund_rows() -> None:
    result = tools.filter_rows_impl(category="money back")

    assert result.match_count == 2
    assert result.row_ids == [0, 3]
    assert result.applied_filters["category"] == "money back"


@pytest.mark.parametrize(
    ("category_alias", "expected_row_ids"),
    [
        ("customer feedback", [2]),
        ("product feedback", [2]),
        ("complaints", [4]),
        ("contact support", [1]),
        ("customer service", [1]),
        ("contact customer service", [1]),
    ],
)
def test_filter_rows_normalizes_non_refund_category_aliases(
    category_alias: str,
    expected_row_ids: list[int],
) -> None:
    result = tools.filter_rows_impl(category=category_alias)

    assert result.match_count == len(expected_row_ids)
    assert result.row_ids == expected_row_ids


def test_filter_rows_filters_by_intent_case_insensitive() -> None:
    result = tools.filter_rows_impl(intent="GET_REFUND")

    assert result.match_count == 1
    assert result.row_ids == [0]


def test_filter_rows_filters_by_text_query() -> None:
    result = tools.filter_rows_impl(text_query="billing")

    assert result.match_count == 1
    assert result.row_ids == [4]


def test_resolve_filter_value_maps_refund_requests_to_refund_category() -> None:
    result = tools.resolve_filter_value_impl(
        query="refund requests",
        columns=["category", "intent"],
        top_k=5,
    )

    assert result.confidence == "high"
    assert result.recommended_filter == {
        "category": "REFUND",
        "intent": None,
    }
    assert result.candidates[0].column == "category"
    assert result.candidates[0].value == "REFUND"
    assert result.candidates[0].count == 2


def test_resolve_filter_value_does_not_switch_explicit_intent_to_category() -> None:
    result = tools.resolve_filter_value_impl(
        query="SHIPPING",
        columns=["intent"],
        top_k=5,
    )

    assert result.confidence == "none"
    assert result.recommended_filter == {
        "category": None,
        "intent": None,
    }
    assert result.candidates == []


def test_resolve_filter_value_maps_shipping_to_category_when_category_allowed() -> None:
    result = tools.resolve_filter_value_impl(
        query="SHIPPING",
        columns=["category"],
        top_k=5,
    )

    assert result.confidence == "high"
    assert result.recommended_filter == {
        "category": "SHIPPING",
        "intent": None,
    }
    assert result.candidates[0].column == "category"
    assert result.candidates[0].value == "SHIPPING"


def test_assignment_cancellation_text_query_finds_cancellation_rows() -> None:
    result = tools.filter_rows_impl(text_query="cancellation")

    assert result.match_count == 1
    assert result.row_ids == [9]


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


def test_assignment_shipping_examples_filter_then_sample() -> None:
    filter_result = tools.filter_rows_impl(category="SHIPPING")
    examples_result = tools.sample_examples_impl(
        row_ids=filter_result.row_ids,
        n=2,
        offset=0,
    )

    assert filter_result.match_count == 2
    assert filter_result.row_ids == [5, 6]
    assert len(examples_result.examples) == 2
    assert examples_result.examples[0].category == "SHIPPING"
    assert examples_result.examples[1].category == "SHIPPING"
    assert examples_result.next_offset == 2


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


def test_assignment_account_intent_distribution_groups_filtered_rows() -> None:
    filter_result = tools.filter_rows_impl(category="ACCOUNT")
    group_result = tools.group_counts_impl(
        group_by="intent",
        row_ids=filter_result.row_ids,
        top_k=10,
    )

    assert filter_result.match_count == 2
    assert filter_result.row_ids == [7, 8]
    assert group_result.group_by == "intent"
    assert group_result.counts == [
        tools.GroupCountRow(label="recover_password", count=1),
        tools.GroupCountRow(label="delete_account", count=1),
    ]


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


def test_summarize_rows_returns_non_empty_summary(
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


def test_schema_based_wrappers_call_implementations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        tools,
        "get_summarizer_llm",
        lambda: (_ for _ in ()).throw(RuntimeError("No summarizer available.")),
    )

    filter_result = tools.filter_rows(
        tools.DatasetFilterInput(category="REFUND"),
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


def test_summarize_rows_falls_back_when_llm_call_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingSummarizerLLM:
        def invoke(self, messages):
            raise ValueError("Simulated provider failure.")

    monkeypatch.setattr(
        tools,
        "get_summarizer_llm",
        lambda: FailingSummarizerLLM(),
    )

    result = tools.summarize_rows_impl(
        row_ids=[0, 3],
        focus="refund requests",
        max_examples=10,
    )

    assert result.row_count_used == 2
    assert "Rows reviewed: 2" in result.summary
    assert "Top categories: REFUND (2)" in result.summary