"""Model factory.

Trade-off: we isolate vendor SDKs here so nodes never import ChatOpenAI
directly. Swapping GPT-4o-mini for a local Ollama model is a config change,
not a graph rewrite. Temperature stays 0 because underwriting scores must be
reproducible; creative decoding would make HITL decisions non-deterministic.
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelProvider(str, Enum):
    OPENAI = "openai"
    OLLAMA = "ollama"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    model_provider: ModelProvider = ModelProvider.OPENAI
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    ollama_model: str = "llama3.1"
    ollama_base_url: str = "http://localhost:11434"
    # HITL threshold is policy, not a model hyperparameter — keep it here
    # so risk scoring and graph routing cannot drift independently.
    human_review_risk_threshold: float = 7.0


@lru_cache
def get_settings() -> Settings:
    return Settings()


def get_chat_model(settings: Settings | None = None) -> BaseChatModel:
    """Return the configured chat model.

    Why a factory instead of a global: tests and local-model experiments
    need a different instance without monkey-patching nodes.
    """
    cfg = settings or get_settings()
    if cfg.model_provider is ModelProvider.OLLAMA:
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=cfg.ollama_model,
            base_url=cfg.ollama_base_url,
            temperature=0,
        )

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=cfg.openai_model,
        api_key=cfg.openai_api_key,
        temperature=0,
    )
