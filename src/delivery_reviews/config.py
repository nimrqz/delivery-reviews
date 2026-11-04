"""Configuração central: seed, nomes de repositórios e variáveis de ambiente."""
from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np

SEED = 42
BASE_MODEL = "neuralmind/bert-base-portuguese-cased"
MAX_LENGTH = 128


def set_seed(seed: int = SEED) -> None:
    """Fixa seeds de random, numpy e (se instalado) torch."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


@dataclass(frozen=True)
class Settings:
    """Variáveis lidas do ambiente (.env)."""

    hf_token: str | None
    hf_username: str | None
    llm_provider: str
    llm_api_key: str | None
    llm_model: str
    llm_base_url: str | None

    @property
    def dataset_repo(self) -> str:
        return f"{self.hf_username}/delivery-reviews"

    @property
    def model_repo(self) -> str:
        return f"{self.hf_username}/bertimbau-delivery-reviews"


def load_settings() -> Settings:
    """Carrega o .env (se existir) e devolve as configurações."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    env = os.environ.get
    return Settings(
        hf_token=env("HF_TOKEN") or None,
        hf_username=env("HF_USERNAME") or None,
        llm_provider=env("LLM_PROVIDER", "anthropic"),
        llm_api_key=env("LLM_API_KEY") or None,
        llm_model=env("LLM_MODEL", "claude-haiku-5-5"),
        llm_base_url=env("LLM_BASE_URL") or None,
    )
