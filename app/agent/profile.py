from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.context import _latest_user_message
from app.agent.llm_factory import get_structured_profile_llm
from app.agent.schemas import ProfileObservationDecision
from app.memory import read_user_profile_impl, update_user_profile_impl
from app.prompts import PROFILE_UPDATE_SYSTEM_PROMPT
from app.state import AgentState


def load_user_profile_node(state: AgentState) -> dict[str, Any]:
    """Load persistent profile memory into graph state."""
    profile = read_user_profile_impl(state["user_id"])
    return {
        "user_profile": profile.profile,
    }


def profile_update_node(state: AgentState) -> dict[str, Any]:
    """Update the persistent profile only when a durable fact is detected."""
    user_query = _latest_user_message(state["messages"])

    if not user_query.strip():
        return {}

    try:
        profile_llm = get_structured_profile_llm()
        decision = profile_llm.invoke(
            [
                SystemMessage(content=PROFILE_UPDATE_SYSTEM_PROMPT),
                HumanMessage(content=user_query),
            ]
        )
        if not isinstance(decision, ProfileObservationDecision):
            decision = ProfileObservationDecision.model_validate(decision)
    except Exception:
        return {}

    observation = decision.observation.strip()
    if not observation:
        return {}

    updated_profile = update_user_profile_impl(
        user_id=state["user_id"],
        new_observation=observation,
    )
    return {
        "user_profile": updated_profile.profile,
    }