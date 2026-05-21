from __future__ import annotations

import pytest

from app import router


class FakeStructuredRouterLLM:
    def __init__(self, route: router.RouteType, reason: str = "test reason") -> None:
        self.route = route
        self.reason = reason
        self.received_messages = None

    def invoke(self, messages):
        self.received_messages = messages
        return router.RouteDecision(route=self.route, reason=self.reason)


@pytest.mark.parametrize(
    ("query", "expected_route"),
    [
        ("How many refund requests?", "structured"),
        ("How many refund requests did we get?", "structured"),
        ("Show 3 examples from REFUND", "structured"),
        ("Show me 5 examples of the SHIPPING category.", "structured"),
        ("What categories exist in the dataset?", "structured"),
        ("What is the distribution of intents in the ACCOUNT category?", "structured"),
        ("Give me a few reimbursement cases", "structured"),
        ("Show me examples of people wanting their money back.", "structured"),
        ("Summarize FEEDBACK", "unstructured"),
        ("Summarize the FEEDBACK category.", "unstructured"),
        ("What are common themes in complaints?", "unstructured"),
        ("Summarize how agents respond to complaint intents.", "unstructured"),
        (
            "How do customer service representatives typically respond to cancellation requests?",
            "unstructured",
        ),
        ("Who is the president?", "out_of_scope"),
        ("What is the weather?", "out_of_scope"),
        ("What's the best CRM software for handling complaints?", "out_of_scope"),
        ("Who is the president of France?", "out_of_scope"),
        ("Who won the 2024 Champions League?", "out_of_scope"),
        ("Write me a poem about customer service.", "out_of_scope"),
    ],
)
def test_route_query_returns_llm_selected_route(
    monkeypatch: pytest.MonkeyPatch,
    query: str,
    expected_route: router.RouteType,
) -> None:
    fake_llm = FakeStructuredRouterLLM(route=expected_route)

    monkeypatch.setattr(router, "get_structured_router_llm", lambda: fake_llm)

    assert router.route_query(query) == expected_route


def test_route_query_with_reason_returns_full_decision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_llm = FakeStructuredRouterLLM(
        route="unstructured",
        reason="The query asks for qualitative dataset themes.",
    )

    monkeypatch.setattr(router, "get_structured_router_llm", lambda: fake_llm)

    result = router.route_query_with_reason("Describe common complaint themes.")

    assert result.route == "unstructured"
    assert result.reason == "The query asks for qualitative dataset themes."


def test_route_query_with_reason_handles_empty_query_without_llm_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_called():
        raise AssertionError("Router LLM should not be called for an empty query.")

    monkeypatch.setattr(router, "get_structured_router_llm", fail_if_called)

    result = router.route_query_with_reason("   ")

    assert result.route == "out_of_scope"
    assert result.reason == "The query is empty."


def test_route_query_with_reason_validates_dict_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DictRouterLLM:
        def invoke(self, messages):
            return {
                "route": "structured",
                "reason": "The query asks for an exact dataset count.",
            }

    monkeypatch.setattr(router, "get_structured_router_llm", lambda: DictRouterLLM())

    result = router.route_query_with_reason("How many rows are in the dataset?")

    assert result == router.RouteDecision(
        route="structured",
        reason="The query asks for an exact dataset count.",
    )


def test_route_query_sends_system_and_human_messages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_llm = FakeStructuredRouterLLM(route="structured")

    monkeypatch.setattr(router, "get_structured_router_llm", lambda: fake_llm)

    router.route_query("How many refund requests?")

    assert fake_llm.received_messages is not None
    assert len(fake_llm.received_messages) == 2
    assert fake_llm.received_messages[0].content == router.ROUTER_SYSTEM_PROMPT
    assert fake_llm.received_messages[1].content == "How many refund requests?"