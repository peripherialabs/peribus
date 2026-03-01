"""
LLM Provider Implementations

Each provider wraps a specific LLM API and provides a uniform
streaming interface for the filesystem.

Multimodal support:
    Messages in config.history may contain a "content_blocks" key
    with a list of ContentBlock objects (from media.py). Each provider
    is responsible for formatting these into its native API format.
    If content_blocks is absent or empty, the plain "content" string is used.
"""

import os
from abc import ABC, abstractmethod
from typing import AsyncIterator, Dict, List, Any, Optional
from dataclasses import dataclass

from .media import (
    ContentBlock, format_content_for_claude,
    format_content_for_openai, format_content_for_gemini
)


@dataclass
class ProviderConfig:
    """Configuration for a provider"""
    model: str = ""
    system: Optional[str] = None
    temperature: float = 1.0
    max_tokens: int = 64000
    max_history: int = 200
    history: List[Dict[str, str]] = None
    
    def __post_init__(self):
        if self.history is None:
            self.history = []


class LLMProvider(ABC):
    """Base class for LLM providers"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name"""
        pass
    
    @property
    def default_model(self) -> str:
        """Default model for this provider. Subclasses can override."""
        suggestions = self.get_models()
        return suggestions[0] if suggestions else ""
    
    def get_models(self) -> List[str]:
        """
        Return list of suggested models.
        
        These are convenience suggestions only — any model string
        is accepted and passed directly to the API.
        """
        return []
    
    @staticmethod
    def _get_content_blocks(msg: Dict[str, Any]) -> Optional[List[ContentBlock]]:
        """Extract content blocks from a history message dict, if present."""
        blocks = msg.get("content_blocks")
        if blocks and isinstance(blocks, list) and len(blocks) > 0:
            # Blocks are already ContentBlock objects (passed from agent)
            if isinstance(blocks[0], ContentBlock):
                return blocks
        return None
    
    @abstractmethod
    async def stream_response(
        self,
        config: ProviderConfig
    ) -> AsyncIterator[str]:
        """
        Stream response from the LLM.
        
        Yields text chunks as they arrive.
        """
        pass

def _build_openai_messages(config: ProviderConfig) -> List[Dict[str, Any]]:
    """
    Build messages list for OpenAI-compatible APIs with multimodal support.
    Used by OpenAI, Groq, OpenRouter, Cerebras, and Moonshot providers.
    """
    messages = []
    
    if config.system:
        messages.append({"role": "system", "content": config.system})
    
    for msg in config.history:
        blocks = LLMProvider._get_content_blocks(msg)
        if blocks:
            content = format_content_for_openai(blocks)
        else:
            content = msg["content"]
        messages.append({"role": msg["role"], "content": content})
    
    return messages


class ClaudeProvider(LLMProvider):
    """Anthropic Claude provider"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not found")
        
        from anthropic import AsyncAnthropic
        self.client = AsyncAnthropic(api_key=self.api_key)
    
    @property
    def name(self) -> str:
        return "claude"
    
    def get_models(self) -> List[str]:
        return [
            "claude-sonnet-4-6",
            "claude-opus-4-6",
            "claude-haiku-4-5-20251001"
        ]
    
    async def stream_response(self, config: ProviderConfig) -> AsyncIterator[str]:
        # Build messages with multimodal content support
        messages = []
        for msg in config.history:
            blocks = self._get_content_blocks(msg)
            if blocks:
                content = format_content_for_claude(blocks)
            else:
                content = msg["content"]
            messages.append({"role": msg["role"], "content": content})
        
        request = {
            "model": config.model,
            "max_tokens": config.max_tokens,
            "messages": messages,
        }
        
        if config.system:
            request["system"] = config.system
        
        async with self.client.messages.stream(**request) as stream:
            async for text in stream.text_stream:
                yield text


class GeminiProvider(LLMProvider):
    """Google Gemini provider"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found")
        
        from google import genai
        self.client = genai.Client(api_key=self.api_key)
    
    @property
    def name(self) -> str:
        return "gemini"
    
    def get_models(self) -> List[str]:
        return [
            "gemini-3-pro-preview",
            "gemini-3-flash-preview",
            "gemini-2.5-pro",
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite"
        ]
    
    async def stream_response(self, config: ProviderConfig) -> AsyncIterator[str]:
        # Convert history to Gemini format with multimodal support
        contents = []
        for msg in config.history:
            role = "user" if msg["role"] == "user" else "model"
            blocks = self._get_content_blocks(msg)
            if blocks:
                parts = format_content_for_gemini(blocks)
            else:
                parts = [{"text": msg["content"]}]
            contents.append({
                "role": role,
                "parts": parts,
            })
        
        # Use the async client (client.aio) for true non-blocking streaming
        response = await self.client.aio.models.generate_content_stream(
            model=config.model,
            contents=contents,
            config={
                "system_instruction": config.system,
                "temperature": config.temperature,
            }
        )
        
        async for chunk in response:
            if chunk.text:
                yield chunk.text


class OpenAIProvider(LLMProvider):
    """OpenAI GPT provider"""
    
    # Models that only support the Responses API (v1/responses),
    # not Chat Completions (v1/chat/completions).
    RESPONSES_ONLY_PATTERNS = ("codex",)
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found")
        
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI(api_key=self.api_key)
    
    @property
    def name(self) -> str:
        return "openai"
    
    def get_models(self) -> List[str]:
        return [
            "gpt-5.3-codex",
            "gpt-5.2-codex",
            "gpt-5.2-2025-12-11",
            "gpt-5-mini-2025-08-07",
            "gpt-5-nano-2025-08-07",
        ]
    
    def _is_responses_model(self, model: str) -> bool:
        """Check if a model requires the Responses API instead of Chat Completions."""
        model_lower = model.lower()
        return any(p in model_lower for p in self.RESPONSES_ONLY_PATTERNS)
    
    async def stream_response(self, config: ProviderConfig) -> AsyncIterator[str]:
        if self._is_responses_model(config.model):
            async for text in self._stream_responses_api(config):
                yield text
        else:
            async for text in self._stream_chat_completions(config):
                yield text
    
    async def _stream_chat_completions(self, config: ProviderConfig) -> AsyncIterator[str]:
        """Stream via the Chat Completions API (v1/chat/completions)."""
        messages = _build_openai_messages(config)
        
        stream = await self.client.chat.completions.create(
            model=config.model,
            messages=messages,
            stream=True,
            max_completion_tokens=config.max_tokens,
        )
        
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    
    async def _stream_responses_api(self, config: ProviderConfig) -> AsyncIterator[str]:
        """Stream via the Responses API (v1/responses) for codex models."""
        # Build input in Responses API format
        input_items = []
        for msg in config.history:
            blocks = self._get_content_blocks(msg)
            if blocks:
                content = format_content_for_openai(blocks)
            else:
                content = msg["content"]
            input_items.append({"role": msg["role"], "content": content})
        
        stream = await self.client.responses.create(
            model=config.model,
            input=input_items,
            stream=True,
            instructions=config.system or "",
        )
        
        async for event in stream:
            if event.type == "response.output_text.delta":
                yield event.delta


class GroqProvider(LLMProvider):
    """Groq provider for fast inference"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found")
        
        from groq import AsyncGroq
        self.client = AsyncGroq(api_key=self.api_key)
    
    @property
    def name(self) -> str:
        return "groq"
    
    def get_models(self) -> List[str]:
        return [
            "moonshotai/kimi-k2-instruct-0905",
            "qwen/qwen3-32b",
            "meta-llama/llama-prompt-guard-2-86m",
            "meta-llama/llama-prompt-guard-2-22m",
            "meta-llama/llama-4-scout-17b-16e-instruct",
            "meta-llama/llama-4-maverick-17b-128e-instruct",
            "openai/gpt-oss-safeguard-20b"
        ]
    
    async def stream_response(self, config: ProviderConfig) -> AsyncIterator[str]:
        messages = _build_openai_messages(config)
        
        stream = await self.client.chat.completions.create(
            model=config.model,
            messages=messages,
            stream=True,
            max_tokens=min(config.max_tokens, 32000),
        )
        
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class CerebrasProvider(LLMProvider):
    """Cerebras Cloud provider for ultra-fast inference"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("CEREBRAS_API_KEY")
        if not self.api_key:
            raise ValueError("CEREBRAS_API_KEY not found")
        
        from cerebras.cloud.sdk import AsyncCerebras
        self.client = AsyncCerebras(api_key=self.api_key)
    
    @property
    def name(self) -> str:
        return "cerebras"
    
    def get_models(self) -> List[str]:
        return [
            "zai-glm-4.7",
            "gpt-oss-120b",
            "qwen-3-235b-a22b-instruct-2507",
            "llama3.1-8b",
        ]
    
    # Per-model max_completion_tokens caps for Cerebras.
    # GLM 4.7 supports up to 128K output but quality degrades at
    # extreme lengths; keep a sensible default cap.
    MODEL_MAX_TOKENS = {
        "zai-glm-4.7": 65000,
        "gpt-oss-120b": 16384,
        "qwen-3-235b-a22b-instruct-2507": 16384,
    }
    DEFAULT_MAX_TOKENS = 8192

    async def stream_response(self, config: ProviderConfig) -> AsyncIterator[str]:
        messages = _build_openai_messages(config)
        
        cap = self.MODEL_MAX_TOKENS.get(config.model, self.DEFAULT_MAX_TOKENS)
        max_tokens = min(config.max_tokens, cap)
        
        # Cerebras uses max_completion_tokens (not max_tokens).
        # For GLM 4.7 reasoning models, also support disable_reasoning.
        kwargs = dict(
            model=config.model,
            messages=messages,
            stream=True,
            max_completion_tokens=max_tokens,
            temperature=config.temperature,
            top_p=0.95,
        )
        
        stream = await self.client.chat.completions.create(**kwargs)
        
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class OpenRouterProvider(LLMProvider):
    """OpenRouter provider for multiple models"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not found")
        
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url="https://openrouter.ai/api/v1"
        )
    
    @property
    def name(self) -> str:
        return "openrouter"
    
    def get_models(self) -> List[str]:
        return [
            "minimax/minimax-m2.5",
            "anthropic/claude-opus-4.6",
            "anthropic/claude-opus-4.5",
            "anthropic/claude-haiku-4.5",
            "anthropic/claude-sonnet-4.5",
            "anthropic/claude-opus-4.1",
            "anthropic/claude-opus-4",
            "anthropic/claude-sonnet-4",
            "openai/gpt-5.2-codex",
            "openai/gpt-5.2-pro",
            "openai/gpt-5.2"
        ]
    
    async def stream_response(self, config: ProviderConfig) -> AsyncIterator[str]:
        messages = _build_openai_messages(config)
        
        stream = await self.client.chat.completions.create(
            model=config.model,
            messages=messages,
            stream=True,
            max_tokens=config.max_tokens,
        )
        
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

class MoonShotAIProvider(LLMProvider):
    """Moonshot AI (Kimi) provider using OpenAI-compatible API"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("MOONSHOT_API_KEY")
        if not self.api_key:
            raise ValueError("MOONSHOT_API_KEY not found")
        
        from openai import AsyncOpenAI
        # Moonshot AI is fully OpenAI-compatible
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url="https://api.moonshot.ai/v1"
        )
    
    @property
    def name(self) -> str:
        return "moonshot"
    
    def get_models(self) -> List[str]:
        return [
            "kimi-k2.5",
            "kimi-k2-instruct",
            "moonshot-v1-8k",
            "moonshot-v1-32k",
            "moonshot-v1-128k",
        ]
    
    async def stream_response(self, config: ProviderConfig) -> AsyncIterator[str]:
        messages = _build_openai_messages(config)
        
        # Moonshot supports temperature [0, 1]
        clamped_temp = max(0.0, min(1.0, config.temperature))
        
        stream = await self.client.chat.completions.create(
            model=config.model,
            messages=messages,
            stream=True,
            max_tokens=config.max_tokens,
            temperature=clamped_temp,
        )
        
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


# Provider registry
_PROVIDERS = {
    "claude": ClaudeProvider,
    "gemini": GeminiProvider,
    "openai": OpenAIProvider,
    "groq": GroqProvider,
    "openrouter": OpenRouterProvider,
    "cerebras": CerebrasProvider,
    "moonshot": MoonShotAIProvider,
}

_provider_instances: Dict[str, LLMProvider] = {}


def get_provider(name: str) -> LLMProvider:
    """
    Get or create a provider instance.
    
    Args:
        name: Provider name (claude, gemini, openai, groq, openrouter)
    
    Returns:
        Provider instance
    """
    if name not in _provider_instances:
        if name not in _PROVIDERS:
            raise ValueError(f"Unknown provider: {name}. Available: {list(_PROVIDERS.keys())}")
        
        try:
            _provider_instances[name] = _PROVIDERS[name]()
        except Exception as e:
            raise ValueError(f"Failed to initialize provider {name}: {e}")
    
    return _provider_instances[name]


def list_providers() -> List[str]:
    """Return list of available provider names"""
    return list(_PROVIDERS.keys())


def register_provider(name: str, provider_class: type):
    """Register a custom provider"""
    _PROVIDERS[name] = provider_class