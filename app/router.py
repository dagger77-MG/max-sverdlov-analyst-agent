from __future__ import annotations

from functools import lru_cache
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
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

3. out_of_scope
Use this when the query is not about:
- the Bitext Customer Service dataset,
- dataset analysis,
- the current conversation context,
- or the saved user profile.

Important rules:
- Do not answer the user question.
- Do not use general knowledge.
- Only classify the route.
- Dataset-related paraphrases should still be routed correctly.
  For example, "reimbursement cases" can be structured if it likely means refund examples.
- If the user asks about external facts, news, weather, politics, sports, recipes, movies, or general knowledge, classify as out_of_scope.
"""


@lru_cache(maxsize=1)
def get_router_llm():
    """Return a cached OpenAI-compatible chat model used by the router."""
    if not settings.nebius_api_key:
        raise RuntimeError(
            "NEBIUS_API_KEY is missing. Add it to your environment or .env file "
            "before using the LLM router."
        )

    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "LLM routing requires 'langchain-openai'. "
            "Install project dependencies before running the router."
        ) from exc

    return ChatOpenAI(
        model=settings.router_model,
        api_key=settings.nebius_api_key,
        base_url=settings.nebius_base_url,
        temperature=0,
        max_tokens=settings.router_max_tokens,
    )


@lru_cache(maxsize=1)
def get_structured_router_llm():
    """Return a cached router model configured for structured RouteDecision output."""
    return get_router_llm().with_structured_output(RouteDecision)


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