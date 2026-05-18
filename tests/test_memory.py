from __future__ import annotations

from pathlib import Path

from app import memory


def test_sanitize_user_id_normalizes_to_safe_directory_name() -> None:
    assert memory.sanitize_user_id(" Max Sverdlov ") == "max_sverdlov"
    assert memory.sanitize_user_id("User@Example.com") == "user_example.com"
    assert memory.sanitize_user_id("___") == "default_user"
    assert memory.sanitize_user_id("") == "default_user"


def test_get_user_profile_path_uses_sanitized_user_id(
    monkeypatch,
    tmp_path: Path,
) -> None:
    class TestSettings:
        default_user_id = "default_user"
        profile_dir = tmp_path

    monkeypatch.setattr(memory, "settings", TestSettings())

    result = memory.get_user_profile_path("Max Sverdlov")

    assert result == tmp_path / "max_sverdlov" / "context.md"


def test_read_user_profile_returns_empty_profile_when_missing(
    monkeypatch,
    tmp_path: Path,
) -> None:
    class TestSettings:
        default_user_id = "default_user"
        profile_dir = tmp_path

        @staticmethod
        def ensure_runtime_dirs() -> None:
            tmp_path.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(memory, "settings", TestSettings())

    result = memory.read_user_profile_impl("max")

    assert result.user_id == "max"
    assert "# User Profile" in result.profile
    assert "- User ID: max" in result.profile
    assert "No durable user facts or preferences have been saved yet." in result.profile


def test_update_user_profile_creates_profile_file(
    monkeypatch,
    tmp_path: Path,
) -> None:
    class TestSettings:
        default_user_id = "default_user"
        profile_dir = tmp_path

        @staticmethod
        def ensure_runtime_dirs() -> None:
            tmp_path.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(memory, "settings", TestSettings())

    result = memory.update_user_profile_impl(
        user_id="max",
        new_observation="User prefers file-by-file code review.",
    )

    profile_path = tmp_path / "max" / "context.md"

    assert result.user_id == "max"
    assert result.updated is True
    assert profile_path.exists()
    assert "- User prefers file-by-file code review." in result.profile
    assert profile_path.read_text(encoding="utf-8") == result.profile


def test_read_user_profile_reads_existing_profile_after_restart(
    monkeypatch,
    tmp_path: Path,
) -> None:
    class TestSettings:
        default_user_id = "default_user"
        profile_dir = tmp_path

        @staticmethod
        def ensure_runtime_dirs() -> None:
            tmp_path.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(memory, "settings", TestSettings())

    memory.update_user_profile_impl(
        user_id="max",
        new_observation="User works on LangGraph educational projects.",
    )

    result = memory.read_user_profile_impl("max")

    assert result.user_id == "max"
    assert "- User works on LangGraph educational projects." in result.profile


def test_update_user_profile_ignores_empty_observation(
    monkeypatch,
    tmp_path: Path,
) -> None:
    class TestSettings:
        default_user_id = "default_user"
        profile_dir = tmp_path

        @staticmethod
        def ensure_runtime_dirs() -> None:
            tmp_path.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(memory, "settings", TestSettings())

    result = memory.update_user_profile_impl(
        user_id="max",
        new_observation="   ",
    )

    assert result.updated is False
    assert "No durable user facts or preferences have been saved yet." in result.profile


def test_update_user_profile_does_not_duplicate_existing_observation(
    monkeypatch,
    tmp_path: Path,
) -> None:
    class TestSettings:
        default_user_id = "default_user"
        profile_dir = tmp_path

        @staticmethod
        def ensure_runtime_dirs() -> None:
            tmp_path.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(memory, "settings", TestSettings())

    first = memory.update_user_profile_impl(
        user_id="max",
        new_observation="User prefers explicit approval before next file.",
    )
    second = memory.update_user_profile_impl(
        user_id="max",
        new_observation="User prefers explicit approval before next file.",
    )

    assert first.updated is True
    assert second.updated is False
    assert second.profile.count("- User prefers explicit approval before next file.") == 1


def test_schema_based_profile_wrappers(
    monkeypatch,
    tmp_path: Path,
) -> None:
    class TestSettings:
        default_user_id = "default_user"
        profile_dir = tmp_path

        @staticmethod
        def ensure_runtime_dirs() -> None:
            tmp_path.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(memory, "settings", TestSettings())

    update_result = memory.update_user_profile(
        memory.UpdateUserProfileInput(
            user_id="max",
            new_observation="User likes visible reasoning traces.",
        )
    )

    read_result = memory.read_user_profile(
        memory.ReadUserProfileInput(user_id="max")
    )

    assert update_result.updated is True
    assert read_result.user_id == "max"
    assert "- User likes visible reasoning traces." in read_result.profile