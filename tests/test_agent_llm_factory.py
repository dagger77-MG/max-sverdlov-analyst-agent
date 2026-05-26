from __future__ import annotations

import sys
import types

import pytest

from app.agent import llm_factory


class FakeSettings:
    nebius_api_key = "test-api-key"
    nebius_base_url = "https://example.test/v1/"
    agent_model = "test-agent-model"
    max_tokens = 3084


class FakeChatOpenAI:
    created_instances = []

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        temperature: int,
        max_tokens: int,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.structured_schema = None
        FakeChatOpenAI.created_instances.append(self)

    def with_structured_output(self, schema):
        self.structured_schema = schema
        return self


@pytest.fixture(autouse=True)
def clear_llm_factory_caches() -> None:
    llm_factory.get_agent_llm.cache_clear()
    llm_factory.get_structured_tool_planner_llm.cache_clear()
    llm_factory.get_structured_observation_reviewer_llm.cache_clear()
    llm_factory.get_structured_profile_llm.cache_clear()
    yield
    llm_factory.get_agent_llm.cache_clear()
    llm_factory.get_structured_tool_planner_llm.cache_clear()
    llm_factory.get_structured_observation_reviewer_llm.cache_clear()
    llm_factory.get_structured_profile_llm.cache_clear()


@pytest.fixture
def fake_langchain_openai(monkeypatch):
    fake_module = types.ModuleType("langchain_openai")
    fake_module.ChatOpenAI = FakeChatOpenAI
    FakeChatOpenAI.created_instances = []

    monkeypatch.setitem(sys.modules, "langchain_openai", fake_module)
    monkeypatch.setattr(llm_factory, "settings", FakeSettings)

    return fake_module


def test_create_agent_chat_llm_uses_openai_compatible_nebius_settings(
    fake_langchain_openai,
) -> None:
    llm = llm_factory._create_agent_chat_llm(max_tokens=123)

    assert isinstance(llm, FakeChatOpenAI)
    assert llm.model == "test-agent-model"
    assert llm.api_key == "test-api-key"
    assert llm.base_url == "https://example.test/v1/"
    assert llm.temperature == 0
    assert llm.max_tokens == 123


def test_create_agent_chat_llm_requires_api_key(monkeypatch) -> None:
    class MissingKeySettings(FakeSettings):
        nebius_api_key = None

    monkeypatch.setattr(llm_factory, "settings", MissingKeySettings)

    with pytest.raises(RuntimeError) as exc_info:
        llm_factory._create_agent_chat_llm(max_tokens=123)

    assert "NEBIUS_API_KEY is missing" in str(exc_info.value)


def test_create_agent_chat_llm_wraps_missing_langchain_openai(monkeypatch) -> None:
    monkeypatch.setattr(llm_factory, "settings", FakeSettings)
    monkeypatch.setitem(sys.modules, "langchain_openai", None)

    with pytest.raises(RuntimeError) as exc_info:
        llm_factory._create_agent_chat_llm(max_tokens=123)

    assert "requires 'langchain-openai'" in str(exc_info.value)


def test_get_agent_llm_uses_configured_max_tokens(fake_langchain_openai) -> None:
    llm = llm_factory.get_agent_llm()

    assert llm.max_tokens == FakeSettings.max_tokens


def test_get_structured_tool_planner_llm_uses_smaller_output_budget(
    fake_langchain_openai,
) -> None:
    llm = llm_factory.get_structured_tool_planner_llm()

    assert llm.max_tokens == 1024
    assert llm.structured_schema is llm_factory.ToolPlanDecision


def test_get_structured_observation_reviewer_llm_uses_reviewer_schema(
    fake_langchain_openai,
) -> None:
    llm = llm_factory.get_structured_observation_reviewer_llm()

    assert llm.structured_schema is llm_factory.ObservationReviewDecision


def test_get_structured_profile_llm_uses_profile_schema(fake_langchain_openai) -> None:
    llm = llm_factory.get_structured_profile_llm()

    assert llm.structured_schema is llm_factory.ProfileObservationDecision