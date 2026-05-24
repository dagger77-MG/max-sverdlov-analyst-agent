from __future__ import annotations

from pathlib import Path

from app.config import AppConfig


def test_app_config_from_env_reads_nebius_api_key(
    monkeypatch,
) -> None:
    monkeypatch.setenv("NEBIUS_API_KEY", "test-api-key")

    config = AppConfig.from_env()

    assert config.nebius_api_key == "test-api-key"


def test_app_config_from_env_uses_expected_project_paths() -> None:
    config = AppConfig.from_env()

    assert config.project_root == Path(__file__).resolve().parents[1]
    assert config.data_dir == config.project_root / "data"
    assert config.profile_dir == config.project_root / ".user_profiles"
    assert config.checkpoint_dir == config.project_root / ".checkpoints"
    assert config.dataset_cache_file == config.data_dir / "bitext_customer_service.csv"


def test_app_config_keeps_model_names_in_config() -> None:
    config = AppConfig.from_env()

    assert config.router_model == "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"
    assert config.agent_model == "nvidia/Llama-3_1-Nemotron-Ultra-253B-v1"


def test_app_config_sets_small_router_output_budget() -> None:
    config = AppConfig.from_env()

    assert config.router_max_tokens == 128
    assert config.router_max_tokens > 0
    assert config.router_max_tokens < config.max_tokens


def test_normalize_max_iterations_uses_default_when_missing() -> None:
    config = AppConfig.from_env()

    assert config.normalize_max_iterations(None) == config.max_iterations


def test_normalize_max_iterations_clamps_to_allowed_range() -> None:
    config = AppConfig.from_env()

    assert config.normalize_max_iterations(1) == config.min_iterations
    assert config.normalize_max_iterations(999) == config.max_allowed_iterations
    assert config.normalize_max_iterations(12) == 12