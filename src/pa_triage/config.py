"""Application settings, loaded from environment / .env.

Settings are intentionally centralized so the provider, model, and data
locations are all swappable without touching agent code.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve project root from this file: src/pa_triage/config.py -> project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"


class Settings(BaseSettings):
    """Runtime configuration.

    Values come from environment variables (or a local ``.env``). Field names
    map to upper-cased env vars, e.g. ``llm_provider`` <- ``LLM_PROVIDER``.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- LLM provider selection ---
    llm_provider: Literal["gemini", "ollama"] = "gemini"
    llm_temperature: float = 0.0

    # Gemini
    google_api_key: str | None = None
    gemini_model: str = "gemini-2.5-flash"

    # Ollama
    ollama_model: str = "qwen2.5"
    ollama_base_url: str = "http://localhost:11434"

    # --- Data locations ---
    data_dir: Path = DATA_DIR
    crosswalk_path: Path = Field(default=DATA_DIR / "crosswalk.json")
    policy_path: Path = Field(default=DATA_DIR / "policy.json")
    samples_dir: Path = Field(default=DATA_DIR / "samples")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
