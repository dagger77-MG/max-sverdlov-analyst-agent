from __future__ import annotations

import pandas as pd

from app.data_loader import (
    add_stable_row_id,
    normalize_column_name,
    normalize_columns,
)


def test_normalize_column_name_converts_unknown_column_to_snake_case() -> None:
    assert normalize_column_name("Customer Message Text") == "customer_message_text"
    assert normalize_column_name("customer-message-text") == "customer_message_text"
    assert normalize_column_name("customer/message/text") == "customer_message_text"


def test_normalize_column_name_applies_known_aliases() -> None:
    assert normalize_column_name("Customer Request") == "instruction"
    assert normalize_column_name("customer-request") == "instruction"
    assert normalize_column_name("customer/request") == "instruction"
    assert normalize_column_name("query") == "instruction"
    assert normalize_column_name("question") == "instruction"
    assert normalize_column_name("answer") == "response"
    assert normalize_column_name("label") == "category"

def test_normalize_columns_renames_dataframe_columns() -> None:
    df = pd.DataFrame(
        {
            "Query": ["I need help"],
            "Answer": ["Sure"],
            "Label": ["SUPPORT"],
        }
    )

    result = normalize_columns(df)

    assert list(result.columns) == ["instruction", "response", "category"]


def test_add_stable_row_id_adds_integer_ids_when_missing() -> None:
    df = pd.DataFrame({"instruction": ["a", "b", "c"]})

    result = add_stable_row_id(df)

    assert "row_id" in result.columns
    assert result["row_id"].tolist() == [0, 1, 2]


def test_add_stable_row_id_preserves_existing_row_id() -> None:
    df = pd.DataFrame(
        {
            "row_id": [10, 11],
            "instruction": ["a", "b"],
        }
    )

    result = add_stable_row_id(df)

    assert result["row_id"].tolist() == [10, 11]