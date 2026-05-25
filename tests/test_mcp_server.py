from __future__ import annotations

from app import mcp_server


def _model_result(**values):
    return type(
        "ModelResult",
        (),
        {
            **values,
            "model_dump": lambda self: values,
        },
    )()


def test_mcp_server_exposes_expected_tools_only() -> None:
    assert hasattr(mcp_server, "get_dataset_schema")
    assert hasattr(mcp_server, "read_user_profile")
    assert hasattr(mcp_server, "resolve_filter_value")
    assert hasattr(mcp_server, "count_rows")
    assert hasattr(mcp_server, "sample_examples")
    assert hasattr(mcp_server, "group_counts")
    assert hasattr(mcp_server, "summarize_rows")

    assert not hasattr(mcp_server, "filter_rows")
    assert not hasattr(mcp_server, "update_user_profile")


def test_read_user_profile_delegates_to_memory_impl(monkeypatch) -> None:
    captured_inputs: list[str] = []

    def fake_read_user_profile_impl(user_id: str):
        captured_inputs.append(user_id)
        return _model_result(
            user_id=user_id,
            profile="# User Profile\n\n- Test profile\n",
        )

    monkeypatch.setattr(
        mcp_server,
        "read_user_profile_impl",
        fake_read_user_profile_impl,
    )

    result = mcp_server.read_user_profile(user_id="max")

    assert captured_inputs == ["max"]
    assert result == {
        "user_id": "max",
        "profile": "# User Profile\n\n- Test profile\n",
    }


def test_get_dataset_schema_delegates_to_tools_impl(monkeypatch) -> None:
    captured_inputs: list[bool] = []

    def fake_get_dataset_schema_impl(include_sample_values: bool = True):
        captured_inputs.append(include_sample_values)
        return _model_result(
            columns=["row_id", "instruction", "response", "category", "intent"],
            row_count=10,
            sample_values=None,
        )

    monkeypatch.setattr(
        mcp_server,
        "get_dataset_schema_impl",
        fake_get_dataset_schema_impl,
    )

    result = mcp_server.get_dataset_schema(include_sample_values=False)

    assert captured_inputs == [False]
    assert result == {
        "columns": ["row_id", "instruction", "response", "category", "intent"],
        "row_count": 10,
        "sample_values": None,
    }


def test_count_rows_delegates_to_tools_impl(monkeypatch) -> None:
    captured_inputs: list[dict] = []

    def fake_count_rows_impl(
        category=None,
        intent=None,
        text_query=None,
    ):
        captured_inputs.append(
            {
                "category": category,
                "intent": intent,
                "text_query": text_query,
            }
        )
        return _model_result(
            count=2992,
            applied_filters={
                "category": category,
                "intent": intent,
                "text_query": text_query,
            },
        )

    monkeypatch.setattr(mcp_server, "count_rows_impl", fake_count_rows_impl)

    result = mcp_server.count_rows(category="REFUND")

    assert captured_inputs == [
        {
            "category": "REFUND",
            "intent": None,
            "text_query": None,
        }
    ]
    assert result == {
        "count": 2992,
        "applied_filters": {
            "category": "REFUND",
            "intent": None,
            "text_query": None,
        },
    }


def test_sample_examples_delegates_to_tools_impl(monkeypatch) -> None:
    captured_inputs: list[dict] = []

    def fake_sample_examples_impl(
        category=None,
        intent=None,
        text_query=None,
        n=3,
        offset=0,
    ):
        captured_inputs.append(
            {
                "category": category,
                "intent": intent,
                "text_query": text_query,
                "n": n,
                "offset": offset,
            }
        )
        return _model_result(
            examples=[],
            next_offset=5,
            match_count=10,
            applied_filters={
                "category": category,
                "intent": intent,
                "text_query": text_query,
            },
        )

    monkeypatch.setattr(
        mcp_server,
        "sample_examples_impl",
        fake_sample_examples_impl,
    )

    result = mcp_server.sample_examples(
        category="SHIPPING",
        n=5,
        offset=0,
    )

    assert captured_inputs == [
        {
            "category": "SHIPPING",
            "intent": None,
            "text_query": None,
            "n": 5,
            "offset": 0,
        }
    ]
    assert result == {
        "examples": [],
        "next_offset": 5,
        "match_count": 10,
        "applied_filters": {
            "category": "SHIPPING",
            "intent": None,
            "text_query": None,
        },
    }


def test_group_counts_delegates_to_tools_impl(monkeypatch) -> None:
    captured_inputs: list[dict] = []

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
                    "count": 2,
                }
            ],
            match_count=2,
            applied_filters={
                "category": category,
                "intent": intent,
                "text_query": text_query,
            },
        )

    monkeypatch.setattr(mcp_server, "group_counts_impl", fake_group_counts_impl)

    result = mcp_server.group_counts(
        group_by="intent",
        category="ACCOUNT",
        top_k=20,
    )

    assert captured_inputs == [
        {
            "group_by": "intent",
            "category": "ACCOUNT",
            "intent": None,
            "text_query": None,
            "top_k": 20,
        }
    ]
    assert result == {
        "group_by": "intent",
        "counts": [
            {
                "label": "recover_password",
                "count": 2,
            }
        ],
        "match_count": 2,
        "applied_filters": {
            "category": "ACCOUNT",
            "intent": None,
            "text_query": None,
        },
    }


def test_resolve_filter_value_delegates_to_tools_impl(monkeypatch) -> None:
    captured_inputs: list[dict] = []

    def fake_resolve_filter_value_impl(
        query,
        columns=None,
        top_k=5,
    ):
        captured_inputs.append(
            {
                "query": query,
                "columns": columns,
                "top_k": top_k,
            }
        )
        return _model_result(
            query=query,
            candidates=[],
            recommended_filter={
                "category": None,
                "intent": None,
            },
            confidence="none",
        )

    monkeypatch.setattr(
        mcp_server,
        "resolve_filter_value_impl",
        fake_resolve_filter_value_impl,
    )

    result = mcp_server.resolve_filter_value(
        query="SHIPPING",
        columns=["intent"],
        top_k=5,
    )

    assert captured_inputs == [
        {
            "query": "SHIPPING",
            "columns": ["intent"],
            "top_k": 5,
        }
    ]
    assert result == {
        "query": "SHIPPING",
        "candidates": [],
        "recommended_filter": {
            "category": None,
            "intent": None,
        },
        "confidence": "none",
    }


def test_summarize_rows_delegates_to_tools_impl(monkeypatch) -> None:
    captured_inputs: list[dict] = []

    def fake_summarize_rows_impl(
        category=None,
        intent=None,
        text_query=None,
        focus="Summarize the selected rows.",
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
            summary="Agents explain cancellation policy.",
            row_count_used=100,
            match_count=950,
            focus=focus,
            target_field=target_field,
            applied_filters={
                "category": category,
                "intent": intent,
                "text_query": text_query,
            },
        )

    monkeypatch.setattr(
        mcp_server,
        "summarize_rows_impl",
        fake_summarize_rows_impl,
    )

    result = mcp_server.summarize_rows(
        category="CANCEL",
        focus="How agents respond to cancellation requests",
        target_field="response",
        max_examples=100,
    )

    assert captured_inputs == [
        {
            "category": "CANCEL",
            "intent": None,
            "text_query": None,
            "focus": "How agents respond to cancellation requests",
            "target_field": "response",
            "max_examples": 100,
        }
    ]
    assert result == {
        "summary": "Agents explain cancellation policy.",
        "row_count_used": 100,
        "match_count": 950,
        "focus": "How agents respond to cancellation requests",
        "target_field": "response",
        "applied_filters": {
            "category": "CANCEL",
            "intent": None,
            "text_query": None,
        },
    }