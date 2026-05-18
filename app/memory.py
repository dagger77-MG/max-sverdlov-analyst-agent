from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel, Field

from app.config import settings


class ReadUserProfileInput(BaseModel):
    user_id: str = Field(description="Persistent user identifier.")


class ReadUserProfileOutput(BaseModel):
    user_id: str
    profile: str


class UpdateUserProfileInput(BaseModel):
    user_id: str = Field(description="Persistent user identifier.")
    new_observation: str = Field(
        description="A concise durable fact or preference to merge into the user profile."
    )


class UpdateUserProfileOutput(BaseModel):
    user_id: str
    updated: bool
    profile: str


def sanitize_user_id(user_id: str) -> str:
    """Convert a user ID into a safe directory name."""
    normalized = user_id.strip().lower()

    if not normalized:
        return settings.default_user_id

    normalized = re.sub(r"[^a-z0-9_.-]+", "_", normalized)
    normalized = normalized.strip("._-")

    return normalized or settings.default_user_id


def get_user_profile_path(user_id: str) -> Path:
    """Return the profile file path for a user."""
    safe_user_id = sanitize_user_id(user_id)
    return settings.profile_dir / safe_user_id / "context.md"


def get_empty_profile_text(user_id: str) -> str:
    """Return the default profile text for a user with no saved profile."""
    return (
        "# User Profile\n\n"
        f"- User ID: {sanitize_user_id(user_id)}\n"
        "- No durable user facts or preferences have been saved yet.\n"
    )


def read_user_profile_impl(user_id: str) -> ReadUserProfileOutput:
    """Read a persistent distilled user profile from disk."""
    settings.ensure_runtime_dirs()

    profile_path = get_user_profile_path(user_id)

    if not profile_path.exists():
        return ReadUserProfileOutput(
            user_id=sanitize_user_id(user_id),
            profile=get_empty_profile_text(user_id),
        )

    return ReadUserProfileOutput(
        user_id=sanitize_user_id(user_id),
        profile=profile_path.read_text(encoding="utf-8"),
    )


def update_user_profile_impl(
    user_id: str,
    new_observation: str,
) -> UpdateUserProfileOutput:
    """Append a durable observation to the user's persistent profile.

    This function intentionally stores distilled facts only. The graph node
    that calls it is responsible for deciding whether an observation is durable
    enough to save.
    """
    settings.ensure_runtime_dirs()

    safe_user_id = sanitize_user_id(user_id)
    observation = new_observation.strip()

    current_profile = read_user_profile_impl(safe_user_id).profile

    if not observation:
        return UpdateUserProfileOutput(
            user_id=safe_user_id,
            updated=False,
            profile=current_profile,
        )

    bullet = f"- {observation}"

    if bullet in current_profile:
        return UpdateUserProfileOutput(
            user_id=safe_user_id,
            updated=False,
            profile=current_profile,
        )

    if "No durable user facts or preferences have been saved yet." in current_profile:
        current_profile = (
            "# User Profile\n\n"
            f"- User ID: {safe_user_id}\n"
        )

    updated_profile = current_profile.rstrip() + "\n" + bullet + "\n"

    profile_path = get_user_profile_path(safe_user_id)
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text(updated_profile, encoding="utf-8")

    return UpdateUserProfileOutput(
        user_id=safe_user_id,
        updated=True,
        profile=updated_profile,
    )


def read_user_profile(input_data: ReadUserProfileInput) -> ReadUserProfileOutput:
    return read_user_profile_impl(user_id=input_data.user_id)


def update_user_profile(input_data: UpdateUserProfileInput) -> UpdateUserProfileOutput:
    return update_user_profile_impl(
        user_id=input_data.user_id,
        new_observation=input_data.new_observation,
    )