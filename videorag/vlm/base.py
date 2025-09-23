import os
import abc
from typing import List, Dict, Any, Optional


class VLM(abc.ABC):
    @abc.abstractmethod
    async def generate(self, prompt: str, images: List[Any] | None = None, videos: List[Any] | None = None) -> str:
        raise NotImplementedError


class VLMConfig(dict):
    pass


def build_vlm(config: VLMConfig) -> VLM:
    provider = (config.get("provider") or "").lower()
    if provider == "bailian":
        from .bailian import BailianVLM
        return BailianVLM(config)
    raise ValueError(f"Unknown VLM provider: {provider}")


