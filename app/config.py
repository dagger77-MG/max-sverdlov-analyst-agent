from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv


@dataclass(frozen=True)
class AppConfig:
    """Central application configuration.

    This project uses Nebius Token Factory through an OpenAI-compatible API.

    Expected .env/environment variable:
        NEBIUS_API_KEY

    Model names are kept in this config class for reproducibility.
    """

    project_root: Path

    data_dir: Path
    profile_dir: Path
    checkpoint_dir: Path

    dataset_name: str
    dataset_cache_file: Path

    default_session_id: str
    default_user_id: str

    max_iterations: int
    min_iterations: int
    max_allowed_iterations: int

    required_analysis_columns: tuple[str, ...]

    nebius_api_key: str | None
    nebius_base_url: str

    router_model: str
    agent_model: str

    debug_trace: bool

    max_tokens: int
    router_max_tokens: int

    @classmethod
    def from_env(cls) -> "AppConfig":
        project_root = Path(__file__).resolve().parents[1]

        data_dir = project_root / "data"
        profile_dir = project_root / ".user_profiles"
        checkpoint_dir = project_root / ".checkpoints"

        return cls(
            project_root=project_root,
            data_dir=data_dir,
            profile_dir=profile_dir,
            checkpoint_dir=checkpoint_dir,
            dataset_name="bitext/Bitext-customer-support-llm-chatbot-training-dataset",
            dataset_cache_file=data_dir / "bitext_customer_service.csv",
            default_session_id="default",
            default_user_id="default_user",
            max_iterations=12,
            min_iterations=10,
            max_allowed_iterations=15,
            required_analysis_columns=("instruction", "response", "category", "intent"),
            nebius_api_key=os.getenv("NEBIUS_API_KEY"),
            nebius_base_url="https://api.tokenfactory.nebius.com/v1/",
            router_model="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
            agent_model="nvidia/Llama-3_1-Nemotron-Ultra-253B-v1",
            debug_trace=True,
            router_max_tokens=256,
            max_tokens=1024,
        )

    def ensure_runtime_dirs(self) -> None:
        """Create local runtime directories used by the agent."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.profile_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def normalize_max_iterations(self, value: int | None) -> int:
        """Clamp max iterations to the assignment-supported range."""
        if value is None:
            return self.max_iterations

        return max(
            self.min_iterations,
            min(self.max_allowed_iterations, value),
        )

load_dotenv()
settings = AppConfig.from_env()