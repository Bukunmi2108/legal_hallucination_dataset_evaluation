import os
from abc import ABC, abstractmethod

from evaluation.config import ModelConfig


class LLMProvider(ABC):
    @abstractmethod
    async def generate(self, system_prompt: str, user_prompt: str) -> str: ...


class OpenAIProvider(LLMProvider):
    def __init__(self, model: str, temperature: float):
        from openai import AsyncOpenAI

        self.client = AsyncOpenAI()
        self.model = model
        self.temperature = temperature

    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
        )
        return response.choices[0].message.content or ""


class AzureOpenAIProvider(LLMProvider):
    def __init__(self, model: str, temperature: float):
        from openai import AsyncAzureOpenAI

        self.client = AsyncAzureOpenAI(
            api_key=os.getenv("AZURE_API_KEY"),
            api_version=os.getenv("AZURE_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_API_BASE", ""),
        )
        self.model = model
        self.temperature = temperature

    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
        )
        return response.choices[0].message.content or ""


class AnthropicProvider(LLMProvider):
    def __init__(self, model: str, temperature: float):
        from anthropic import AsyncAnthropic

        self.client = AsyncAnthropic()
        self.model = model
        self.temperature = temperature

    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=self.temperature,
        )
        return response.content[0].text  # type: ignore[union-attr]


class GoogleProvider(LLMProvider):
    def __init__(self, model: str, temperature: float):
        from google import genai

        self.client = genai.Client()
        self.model = model
        self.temperature = temperature

    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        from google.genai import types

        response = await self.client.aio.models.generate_content(
            model=self.model,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=self.temperature,
            ),
        )
        return response.text or ""


_PROVIDER_MAP = {
    "openai": OpenAIProvider,
    "azure": AzureOpenAIProvider,
    "anthropic": AnthropicProvider,
    "google": GoogleProvider,
}


def create_provider(config: ModelConfig) -> LLMProvider:
    cls = _PROVIDER_MAP[config.provider]
    return cls(model=config.api_model_name, temperature=config.temperature)
