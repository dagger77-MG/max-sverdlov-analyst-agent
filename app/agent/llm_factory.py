from __future__ import annotations

from functools import lru_cache

from app.agent.schemas import (
    ObservationReviewDecision,
    ProfileObservationDecision,
    ToolPlanDecision,
)
from app.config import settings


def _create_agent_chat_llm(max_tokens: int):
    """Create an OpenAI-compatible chat model with an explicit output budget."""
    if not settings.nebius_api_key:
        raise RuntimeError(
            "NEBIUS_API_KEY is missing. Add it to your environment or .env file "
            "before using the graph agent."
        )

    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The graph agent requires 'langchain-openai'. "
            "Install project dependencies before running the agent."
        ) from exc

    return ChatOpenAI(
        model=settings.agent_model,
        api_key=settings.nebius_api_key,
        base_url=settings.nebius_base_url,
        temperature=0,
        max_tokens=max_tokens,
        # extra_body={
        #     "enable_thinking": False,
        #     "thinking_budget": 0,
        # },
    )


@lru_cache(maxsize=1)
def get_agent_llm():
    """Return a cached OpenAI-compatible chat model for the data agent."""
    return _create_agent_chat_llm(max_tokens=settings.max_tokens)


@lru_cache(maxsize=1)
def get_structured_tool_planner_llm():
    """Return a cached model configured for next-tool planning decisions."""
    return _create_agent_chat_llm(
        max_tokens=min(settings.max_tokens, 512),
    ).with_structured_output(ToolPlanDecision)


@lru_cache(maxsize=1)
def get_structured_observation_reviewer_llm():
    """Return a cached model configured for observation-readiness decisions."""
    return get_agent_llm().with_structured_output(ObservationReviewDecision)


@lru_cache(maxsize=1)
def get_structured_profile_llm():
    """Return a cached model configured for profile-update decisions."""
    return get_agent_llm().with_structured_output(ProfileObservationDecision)