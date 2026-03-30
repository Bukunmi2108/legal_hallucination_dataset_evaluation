from dataclasses import dataclass
from pathlib import Path
from typing import Literal

SYSTEM_PROMPT = (
    "You are a legal research assistant. Answer the user's legal question "
    "thoroughly, citing specific case names, statute references, and other "
    "legal authorities where relevant. Provide full citations including year, "
    "court, and neutral citation numbers where available."
)

OUTPUT_DIR = Path("data/output")


@dataclass(frozen=True)
class ModelConfig:
    model_id: str
    provider: Literal["openai", "azure", "anthropic", "google"]
    api_model_name: str
    max_concurrency: int = 10
    temperature: float = 0.7


MODEL_REGISTRY: dict[str, ModelConfig] = {
    # GPT-4.1 (baseline / older generation)
    "gpt-4.1": ModelConfig("gpt-4.1", "openai", "gpt-4.1", max_concurrency=5),
    # GPT-5.x family (progression)
    "gpt-5.1": ModelConfig("gpt-5.1", "openai", "gpt-5.1", max_concurrency=5),
    "gpt-5.2": ModelConfig("gpt-5.2", "openai", "gpt-5.2", max_concurrency=5),
    "gpt-5.4": ModelConfig("gpt-5.4", "openai", "gpt-5.4", max_concurrency=3),
    "gpt-5.4-mini": ModelConfig("gpt-5.4-mini", "openai", "gpt-5.4-mini", max_concurrency=10),
    "gpt-5.4-nano": ModelConfig("gpt-5.4-nano", "openai", "gpt-5.4-nano", max_concurrency=15),
}
