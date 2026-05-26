from __future__ import annotations

from typing import Any
import logging

from langchain_core.messages import HumanMessage

from app.agent import profile


def _state(query: str = "I prefer file-by-file implementation review.") -> dict[str, Any]:
    return {
        "messages": [HumanMessage(content=query)],
        "session_id": "test_session",
        "user_id": "max",
        "route": None,
        "route_reason": None,
        "tool_trace": [],
        "last_structured_results": [],
        "user_profile": "",
        "max_iterations": 12,
        "final_answer": None,
    }


def _profile_result(user_id: str, profile_text: str):
    return type(
        "ProfileResult",
        (),
        {
            "user_id": user_id,
            "profile": profile_text,
        },
    )()


class FakeProfileLLM:
    def __init__(self, observation: str) -> None:
        self.observation = observation
        self.received_messages = None

    def invoke(self, messages):
        self.received_messages = messages
        return profile.ProfileObservationDecision(observation=self.observation)


def test_load_user_profile_node_reads_profile(monkeypatch) -> None:
    monkeypatch.setattr(
        profile,
        "read_user_profile_impl",
        lambda user_id: _profile_result(
            user_id,
            "# User Profile\n\n- Test profile\n",
        ),
    )

    result = profile.load_user_profile_node(_state())

    assert result == {
        "user_profile": "# User Profile\n\n- Test profile\n",
    }


def test_profile_update_node_ignores_empty_user_query(monkeypatch) -> None:
    def fail_if_profile_llm_called():
        raise AssertionError("Profile LLM should not be called for empty query.")

    monkeypatch.setattr(
        profile,
        "get_structured_profile_llm",
        fail_if_profile_llm_called,
    )

    result = profile.profile_update_node(_state("   "))

    assert result == {}


def test_profile_update_node_ignores_empty_observation(monkeypatch) -> None:
    fake_llm = FakeProfileLLM(observation="")

    def fail_if_profile_updated(user_id: str, new_observation: str):
        raise AssertionError("Empty observations should not update the profile.")

    monkeypatch.setattr(
        profile,
        "get_structured_profile_llm",
        lambda: fake_llm,
    )
    monkeypatch.setattr(
        profile,
        "update_user_profile_impl",
        fail_if_profile_updated,
    )

    result = profile.profile_update_node(_state())

    assert result == {}
    assert fake_llm.received_messages is not None


def test_profile_update_node_saves_durable_observation(monkeypatch) -> None:
    fake_llm = FakeProfileLLM(
        observation="User prefers file-by-file implementation review."
    )

    captured_update: dict[str, str] = {}

    def fake_update_user_profile_impl(user_id: str, new_observation: str):
        captured_update["user_id"] = user_id
        captured_update["new_observation"] = new_observation
        return _profile_result(
            user_id,
            f"# User Profile\n\n- {new_observation}\n",
        )

    monkeypatch.setattr(
        profile,
        "get_structured_profile_llm",
        lambda: fake_llm,
    )
    monkeypatch.setattr(
        profile,
        "update_user_profile_impl",
        fake_update_user_profile_impl,
    )

    result = profile.profile_update_node(_state())

    assert captured_update == {
        "user_id": "max",
        "new_observation": "User prefers file-by-file implementation review.",
    }
    assert result == {
        "user_profile": (
            "# User Profile\n\n"
            "- User prefers file-by-file implementation review.\n"
        ),
    }


def test_profile_update_node_logs_and_skips_on_decision_failure(
    monkeypatch,
    caplog,
) -> None:
    class FailingProfileLLM:
        def invoke(self, messages):
            raise RuntimeError("Simulated profile decision failure.")

    monkeypatch.setattr(
        profile,
        "get_structured_profile_llm",
        lambda: FailingProfileLLM(),
    )

    caplog.set_level(logging.ERROR, logger=profile.__name__)

    result = profile.profile_update_node(_state())

    assert result == {}
    assert "Profile update decision failed; skipping profile update." in caplog.text
    assert "Simulated profile decision failure." in caplog.text