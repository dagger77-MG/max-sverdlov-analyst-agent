from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from app import router


class FakeStructuredRouterLLM:
    def __init__(self, route: router.RouteType, reason: str = "test reason") -> None:
        self.route = route
        self.reason = reason
        self.received_messages = None

    def invoke(self, messages):
        self.received_messages = messages
        return router.RouteDecision(route=self.route, reason=self.reason)


class FailingStructuredRouterLLM:
    def invoke(self, messages):
        raise ValueError("Simulated structured-output failure.")


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
        ("What do you know about me?", "structured"),
        ("What do you remember about me?", "structured"),
        ("Summarize FEEDBACK", "unstructured"),
        ("Summarize the FEEDBACK category.", "unstructured"),
        ("What are common themes in complaints?", "unstructured"),
        ("Summarize how agents respond to complaint intents.", "unstructured"),
        ("What is the recommended answer for complaint?", "unstructured"),
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


def test_parse_route_decision_accepts_reasoning_text_before_json() -> None:
    raw_output = (
        "We need to classify the user query. This is asking for a recommended "
        "answer for complaint, so it is qualitative dataset analysis.\n\n"
        "{\n"
        '  "route": "unstructured",\n'
        '  "reason": "The query asks for a recommended support response pattern."\n'
        "}"
    )

    result = router._parse_route_decision(raw_output)

    assert result == router.RouteDecision(
        route="unstructured",
        reason="The query asks for a recommended support response pattern.",
    )


def test_parse_route_decision_accepts_profile_reasoning_with_final_json() -> None:
    raw_output = (
        'We need to classify the user query "What do you know about me?" '
        "According to instructions, this is a question about the saved user "
        "profile, so it should be structured. "
        'So output: {"route":"structured","reason":"question about saved user profile"}\n\n'
        '{"route":"structured","reason":"question about saved user profile"}'
    )

    result = router._parse_route_decision(raw_output)

    assert result == router.RouteDecision(
        route="structured",
        reason="question about saved user profile",
    )


def test_extract_embedded_json_object_returns_final_json_object() -> None:
    raw_output = (
        'Example JSON: {"route":"out_of_scope","reason":"example only"}\n'
        '{"route":"structured","reason":"final answer"}'
    )

    assert router._extract_embedded_json_object(raw_output) == (
        '{"route":"structured","reason":"final answer"}'
    )


def test_route_query_with_reason_wraps_router_llm_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        router,
        "get_structured_router_llm",
        lambda: FailingStructuredRouterLLM(),
    )

    with pytest.raises(RuntimeError) as exc_info:
        router.route_query_with_reason("show me 2 more")

    assert "Router failed to produce a valid route decision." in str(exc_info.value)
    assert "ValueError" in str(exc_info.value)
    assert "Simulated structured-output failure." in str(exc_info.value)


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


def test_build_responses_payload_requests_json_object_format() -> None:
    payload = router._build_responses_payload(
        [
            SystemMessage(content=router.ROUTER_SYSTEM_PROMPT),
            HumanMessage(content="How many refund requests?"),
        ]
    )

    assert payload["model"] == router.settings.router_model
    assert payload["instructions"] == router.ROUTER_SYSTEM_PROMPT.strip()
    assert payload["input"] == "How many refund requests?"
    assert payload["temperature"] == 0
    assert payload["max_output_tokens"] == router.settings.router_max_tokens
    assert payload["parallel_tool_calls"] is False
    assert payload["tool_choice"] == "none"
    assert payload["stream"] is False
    assert payload["store"] is False

    text_format = payload["text"]["format"]
    assert text_format == {"type": "json_object"}


def test_extract_response_text_prefers_top_level_output_text() -> None:
    result = router._extract_response_text(
        {
            "output_text": (
                '{"route":"structured","reason":"The query asks for a count."}'
            ),
        }
    )

    assert result == '{"route":"structured","reason":"The query asks for a count."}'


def test_extract_response_text_falls_back_to_nested_output_content() -> None:
    result = router._extract_response_text(
        {
            "output": [
                {
                    "content": [
                        {
                            "type": "output_text",
                            "text": '{"route":"unstructured",',
                        },
                        {
                            "type": "output_text",
                            "text": '"reason":"The query asks for themes."}',
                        },
                    ],
                }
            ],
        }
    )

    assert result == (
        '{"route":"unstructured","reason":"The query asks for themes."}'
    )


def test_extract_response_text_raises_for_incomplete_response() -> None:
    with pytest.raises(ValueError) as exc_info:
        router._extract_response_text(
            {
                "status": "incomplete",
                "incomplete_details": {
                    "reason": "max_output_tokens",
                },
                "usage": {
                    "output_tokens": 256,
                },
            }
        )

    error_text = str(exc_info.value)
    assert "Nebius router response was not complete." in error_text
    assert "max_output_tokens" in error_text
    assert "usage=" in error_text


def test_nebius_responses_router_llm_parses_valid_json_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        router,
        "_post_responses_request",
        lambda payload: {
            "output_text": (
                '{"route":"structured","reason":"The query asks for examples."}'
            ),
        },
    )

    result = router.NebiusResponsesRouterLLM().invoke(
        [
            SystemMessage(content=router.ROUTER_SYSTEM_PROMPT),
            HumanMessage(content="Show me 3 examples from REFUND."),
        ]
    )

    assert result == router.RouteDecision(
        route="structured",
        reason="The query asks for examples.",
    )