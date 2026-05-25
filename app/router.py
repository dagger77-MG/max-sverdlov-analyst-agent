from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, Literal
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.config import settings

RouteType = Literal["structured", "unstructured", "out_of_scope"]


class RouteDecision(BaseModel):
    """Router output for a user query."""

    route: RouteType = Field(
        description=(
            "Query route. Use 'structured' for exact dataset operations, "
            "'unstructured' for qualitative dataset analysis, and "
            "'out_of_scope' for questions unrelated to the dataset/profile."
        )
    )
    reason: str = Field(
        description="Brief explanation of why this route was selected."
    )


ROUTER_SYSTEM_PROMPT = """You are a router for a Bitext Customer Service dataset analyst agent.

Your only task is to classify the user's query into exactly one route:

1. structured
Use this when the user asks for exact dataset operations over the Bitext Customer Service dataset, including:
- counts
- filters
- examples
- samples
- group counts
- category or intent distributions
- most common category or intent
- row counts
- follow-up requests like "show me more", "what about refunds?", or "total of the last two"
- questions about the saved user profile, such as "what do you remember about me?"

2. unstructured
Use this when the user asks for qualitative analysis over Bitext Customer Service dataset rows, including:
- summaries
- themes
- patterns
- common customer requests
- tone analysis
- customer pain points
- interpretation of categories or intents
- recommended/support response patterns, such as a recommended answer for complaints

3. out_of_scope
Use this when the query is not about:
- the Bitext Customer Service dataset,
- dataset analysis,
- the current conversation context,
- or the saved user profile.

Rules:
- Do not answer the user question.
- Do not use general knowledge.
- Only classify the route.
- Dataset-related paraphrases should still be routed correctly.
  For example, "reimbursement cases" can be structured if it likely means refund examples.
- If the user asks about external facts, news, weather, politics, sports, recipes, movies, or general knowledge, classify as out_of_scope.

- Return exactly one JSON object and no other text:
  {"route":"structured|unstructured|out_of_scope","reason":"short reason"}
- Do not include reasoning, markdown, prose, or text before/after the JSON.
"""


class NebiusResponsesRouterLLM:
    """Small Responses API client dedicated to route classification."""

    def invoke(self, messages: list[BaseMessage]) -> RouteDecision:
        payload = _build_responses_payload(messages)
        response_body = _post_responses_request(payload)
        output_text = _extract_response_text(response_body)
        return _parse_route_decision(output_text)


def _parse_route_decision(output_text: str) -> RouteDecision:
    """Parse a RouteDecision, tolerating accidental prose around the JSON."""
    stripped_output = output_text.strip()
    candidates = [stripped_output]

    embedded_json = _extract_embedded_json_object(stripped_output)
    if embedded_json and embedded_json not in candidates:
        candidates.append(embedded_json)

    last_error: Exception | None = None
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return RouteDecision.model_validate(json.loads(candidate))
        except Exception as exc:
            last_error = exc

    raise ValueError(
        "Router response was not valid RouteDecision JSON. "
        f"Raw output: {output_text!r}"
    ) from last_error


def _extract_embedded_json_object(text: str) -> str | None:
    """Return the final JSON-looking object from a text response."""
    end = text.rfind("}")
    if end == -1:
        return None

    start = text.rfind("{", 0, end + 1)
    if start == -1 or start >= end:
        return None

    return text[start: end + 1]


def _messages_to_instructions_and_input(
    messages: list[BaseMessage],
) -> tuple[str, str]:
    """Split LangChain-style messages into Responses API instructions/input."""
    instruction_parts: list[str] = []
    input_parts: list[str] = []

    for message in messages:
        content = str(message.content)
        if isinstance(message, SystemMessage):
            instruction_parts.append(content)
        elif isinstance(message, HumanMessage):
            input_parts.append(content)
        else:
            input_parts.append(content)

    instructions = "\n\n".join(instruction_parts).strip()
    user_input = "\n\n".join(input_parts).strip()

    return instructions, user_input


def _build_responses_payload(messages: list[BaseMessage]) -> dict[str, Any]:
    """Build a Nebius Responses API request for route classification."""
    instructions, user_input = _messages_to_instructions_and_input(messages)

    return {
        "model": settings.router_model,
        "instructions": instructions,
        "input": user_input,
        "temperature": 0,
        "text": {
            "format": {
                "type": "json_object",
            },
        },
        "max_output_tokens": settings.router_max_tokens,
        "parallel_tool_calls": False,
        "tool_choice": "none",
        "stream": False,
        "store": False,
    }


def _responses_endpoint_url() -> str:
    """Return the configured Nebius Responses API endpoint URL."""
    return urljoin(settings.nebius_base_url.rstrip("/") + "/", "responses")


def _post_responses_request(payload: dict[str, Any]) -> dict[str, Any]:
    """POST one request to the Nebius Responses API using only stdlib HTTP."""
    if not settings.nebius_api_key:
        raise RuntimeError(
            "NEBIUS_API_KEY is missing. Add it to your environment or .env file "
            "before using the LLM router."
        )

    request = Request(
        url=_responses_endpoint_url(),
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {settings.nebius_api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=60) as response:
            response_text = response.read().decode("utf-8")
    except HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Nebius router request failed with HTTP {exc.code}: {error_body}"
        ) from exc
    except URLError as exc:
        raise RuntimeError(f"Nebius router request failed: {exc}") from exc

    return json.loads(response_text)


def _response_failure_details(response_body: dict[str, Any]) -> str:
    """Return compact diagnostic details for failed/incomplete responses."""
    status = response_body.get("status")
    error = response_body.get("error")
    incomplete_details = response_body.get("incomplete_details")
    usage = response_body.get("usage")

    return (
        f"status={status!r}; "
        f"error={error!r}; "
        f"incomplete_details={incomplete_details!r}; "
        f"usage={usage!r}"
    )


def _extract_response_text(response_body: dict[str, Any]) -> str:
    """Extract output text from a Responses API response body."""
    if response_body.get("error") or response_body.get("incomplete_details"):
        raise ValueError(
            "Nebius router response was not complete. "
            f"{_response_failure_details(response_body)}"
        )

    output_text = response_body.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    text_parts: list[str] = []
    for output_item in response_body.get("output", []):
        if not isinstance(output_item, dict):
            continue
        for content_item in output_item.get("content", []):
            if not isinstance(content_item, dict):
                continue
            text = content_item.get("text")
            if isinstance(text, str) and text:
                text_parts.append(text)

    combined_text = "".join(text_parts).strip()
    if combined_text:
        return combined_text

    raise ValueError(
        "Nebius router response did not contain output text. "
        f"{_response_failure_details(response_body)}"
    )


@lru_cache(maxsize=1)
def get_structured_router_llm() -> NebiusResponsesRouterLLM:
    """Return a cached router client."""
    return NebiusResponsesRouterLLM()


def route_query_with_reason(query: str) -> RouteDecision:
    """Classify a user query with the LLM router."""
    normalized_query = query.strip()

    if not normalized_query:
        return RouteDecision(
            route="out_of_scope",
            reason="The query is empty.",
        )

    structured_llm = get_structured_router_llm()

    try:
        decision = structured_llm.invoke(
            [
                SystemMessage(content=ROUTER_SYSTEM_PROMPT),
                HumanMessage(content=normalized_query),
            ]
        )

        if isinstance(decision, RouteDecision):
            return decision

        return RouteDecision.model_validate(decision)
    except Exception as exc:
        raise RuntimeError(
            "Router failed to produce a valid route decision. "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc


def route_query(query: str) -> RouteType:
    """Classify a user query before the agent chooses tools."""
    return route_query_with_reason(query).route
