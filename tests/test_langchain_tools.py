from __future__ import annotations

from app import langchain_tools
import pytest


def test_get_dataset_schema_tool_calls_impl(monkeypatch) -> None:
    captured: dict = {}

    def fake_get_dataset_schema_impl(include_sample_values: bool = True):
        captured["include_sample_values"] = include_sample_values
        return type(
            "DatasetSchemaResult",
            (),
            {
                "model_dump": lambda self: {
                    "columns": ["row_id", "instruction", "category"],
                    "row_count": 3,
                    "sample_values": {"category": ["REFUND"]},
                }
            },
        )()

    monkeypatch.setattr(
        langchain_tools,
        "get_dataset_schema_impl",
        fake_get_dataset_schema_impl,
    )

    result = langchain_tools.get_dataset_schema_tool.invoke(
        {"include_sample_values": False}
    )

    assert captured == {"include_sample_values": False}
    assert result == {
        "columns": ["row_id", "instruction", "category"],
        "row_count": 3,
        "sample_values": {"category": ["REFUND"]},
    }


def test_filter_rows_tool_calls_impl(monkeypatch) -> None:
    captured: dict = {}

    def fake_filter_rows_impl(
        category=None,
        intent=None,
        text_query=None,
        limit=None,
    ):
        captured.update(
            {
                "category": category,
                "intent": intent,
                "text_query": text_query,
                "limit": limit,
            }
        )
        return type(
            "FilterRowsResult",
            (),
            {
                "model_dump": lambda self: {
                    "row_ids": [1, 2, 3],
                    "match_count": 3,
                    "applied_filters": captured.copy(),
                }
            },
        )()

    monkeypatch.setattr(
        langchain_tools,
        "filter_rows_impl",
        fake_filter_rows_impl,
    )

    result = langchain_tools.filter_rows_tool.invoke(
        {
            "category": "REFUND",
            "intent": "get_refund",
            "text_query": "refund",
            "limit": 10,
        }
    )

    assert captured == {
        "category": "REFUND",
        "intent": "get_refund",
        "text_query": "refund",
        "limit": 10,
    }
    assert result["row_ids"] == [1, 2, 3]
    assert result["match_count"] == 3


def test_count_rows_tool_calls_impl(monkeypatch) -> None:
    captured: dict = {}

    def fake_count_rows_impl(row_ids=None):
        captured["row_ids"] = row_ids
        return type(
            "CountRowsResult",
            (),
            {"model_dump": lambda self: {"count": len(row_ids or [])}},
        )()

    monkeypatch.setattr(
        langchain_tools,
        "count_rows_impl",
        fake_count_rows_impl,
    )

    result = langchain_tools.count_rows_tool.invoke({"row_ids": [10, 11]})

    assert captured == {"row_ids": [10, 11]}
    assert result == {"count": 2}


def test_sample_examples_tool_calls_impl(monkeypatch) -> None:
    captured: dict = {}

    def fake_sample_examples_impl(row_ids=None, n=3, offset=0):
        captured.update(
            {
                "row_ids": row_ids,
                "n": n,
                "offset": offset,
            }
        )
        return type(
            "SampleExamplesResult",
            (),
            {
                "model_dump": lambda self: {
                    "examples": [
                        {
                            "row_id": 10,
                            "instruction": "Where is my refund?",
                            "response": "Check your account.",
                            "category": "REFUND",
                            "intent": "check_refund_status",
                        }
                    ],
                    "next_offset": 4,
                }
            },
        )()

    monkeypatch.setattr(
        langchain_tools,
        "sample_examples_impl",
        fake_sample_examples_impl,
    )

    result = langchain_tools.sample_examples_tool.invoke(
        {
            "row_ids": [10, 11, 12],
            "n": 1,
            "offset": 3,
        }
    )

    assert captured == {
        "row_ids": [10, 11, 12],
        "n": 1,
        "offset": 3,
    }
    assert result["examples"][0]["row_id"] == 10
    assert result["next_offset"] == 4


def test_group_counts_tool_calls_impl(monkeypatch) -> None:
    captured: dict = {}

    def fake_group_counts_impl(group_by, row_ids=None, top_k=20):
        captured.update(
            {
                "group_by": group_by,
                "row_ids": row_ids,
                "top_k": top_k,
            }
        )
        return type(
            "GroupCountsResult",
            (),
            {
                "model_dump": lambda self: {
                    "group_by": group_by,
                    "counts": [{"label": "REFUND", "count": 3}],
                }
            },
        )()

    monkeypatch.setattr(
        langchain_tools,
        "group_counts_impl",
        fake_group_counts_impl,
    )

    result = langchain_tools.group_counts_tool.invoke(
        {
            "group_by": "category",
            "row_ids": [1, 2, 3],
            "top_k": 5,
        }
    )

    assert captured == {
        "group_by": "category",
        "row_ids": [1, 2, 3],
        "top_k": 5,
    }
    assert result == {
        "group_by": "category",
        "counts": [{"label": "REFUND", "count": 3}],
    }


def test_group_counts_tool_rejects_invalid_group_by() -> None:
    with pytest.raises(Exception) as exc_info:
        langchain_tools.group_counts_tool.invoke(
            {
                "group_by": "invalid",
                "top_k": 5,
            }
        )

    assert "Input should be 'category' or 'intent'" in str(exc_info.value)


def test_summarize_rows_tool_calls_impl(monkeypatch) -> None:
    captured: dict = {}

    def fake_summarize_rows_impl(row_ids, focus, max_examples=100):
        captured.update(
            {
                "row_ids": row_ids,
                "focus": focus,
                "max_examples": max_examples,
            }
        )
        return type(
            "SummarizeRowsResult",
            (),
            {
                "model_dump": lambda self: {
                    "summary": "Refund rows mostly ask about refund status.",
                    "row_count_used": 2,
                    "focus": focus,
                }
            },
        )()

    monkeypatch.setattr(
        langchain_tools,
        "summarize_rows_impl",
        fake_summarize_rows_impl,
    )

    result = langchain_tools.summarize_rows_tool.invoke(
        {
            "row_ids": [10, 11],
            "focus": "refund themes",
            "max_examples": 50,
        }
    )

    assert captured == {
        "row_ids": [10, 11],
        "focus": "refund themes",
        "max_examples": 50,
    }
    assert result == {
        "summary": "Refund rows mostly ask about refund status.",
        "row_count_used": 2,
        "focus": "refund themes",
    }


def test_read_user_profile_tool_calls_impl(monkeypatch) -> None:
    captured: dict = {}

    def fake_read_user_profile_impl(user_id: str):
        captured["user_id"] = user_id
        return type(
            "ReadUserProfileResult",
            (),
            {
                "model_dump": lambda self: {
                    "user_id": user_id,
                    "profile": "# User Profile\n\n- User likes traces.\n",
                }
            },
        )()

    monkeypatch.setattr(
        langchain_tools,
        "read_user_profile_impl",
        fake_read_user_profile_impl,
    )

    result = langchain_tools.read_user_profile_tool.invoke({"user_id": "max"})

    assert captured == {"user_id": "max"}
    assert result == {
        "user_id": "max",
        "profile": "# User Profile\n\n- User likes traces.\n",
    }


def test_exported_tool_lists_include_expected_tools() -> None:
    dataset_tool_names = {
        tool.name for tool in langchain_tools.DATASET_LANGCHAIN_TOOLS
    }
    all_tool_names = {tool.name for tool in langchain_tools.LANGCHAIN_TOOLS}

    assert dataset_tool_names == {
        "get_dataset_schema",
        "filter_rows",
        "count_rows",
        "sample_examples",
        "group_counts",
        "summarize_rows",
    }
    assert all_tool_names == {
        "get_dataset_schema",
        "filter_rows",
        "count_rows",
        "sample_examples",
        "group_counts",
        "summarize_rows",
        "read_user_profile",
    }