from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
PROFILE_DIR = PROJECT_ROOT / ".user_profiles"
CHECKPOINT_DIR = PROJECT_ROOT / ".checkpoints"

DATASET_NAME = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
DATASET_CACHE_FILE = DATA_DIR / "bitext_customer_service.csv"

DEFAULT_SESSION_ID = "default"
DEFAULT_USER_ID = "default_user"

MAX_ITERATIONS = 12
MIN_ITERATIONS = 10
MAX_ALLOWED_ITERATIONS = 15

REQUIRED_ANALYSIS_COLUMNS = ("instruction", "response", "category", "intent")


def ensure_runtime_dirs() -> None:
    """Create local runtime directories used by the agent."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def normalize_max_iterations(value: int | None) -> int:
    """Clamp max iterations to the assignment-supported range."""
    if value is None:
        return MAX_ITERATIONS

    return max(MIN_ITERATIONS, min(MAX_ALLOWED_ITERATIONS, value))