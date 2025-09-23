import os
import httpx
import base64
from typing import List, Any

from .base import VLM, VLMConfig
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from openai import APIConnectionError, RateLimitError
from typing import TYPE_CHECKING
from io import BytesIO
if TYPE_CHECKING:
    import PIL.Image


class BailianVLM(VLM):
    def __init__(self, config: VLMConfig):
        # API key: prefer config -> env -> provided fallback
        self.api_key = config.get("api_key")
        if not self.api_key:
            raise ValueError("BAILIAN_API_KEY is required for bailian provider")

        # Base URL: prefer config -> env -> default
        self.base_url = config.get("base_url")

        # Model & params
        self.model = config.get("model")
        self.max_tokens = config.get("max_tokens", 1024)
        self.temperature = config.get("temperature", 0.7)
        self.timeout = config.get("timeout", 60.0)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
    )
    async def generate(self, prompt: str, images: List["PIL.Image.Image"] | None = None, videos: List[Any] | None = None) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # Build message content
        message_content = []
        if prompt:
            message_content.append({"type": "text", "text": prompt})

        if images:
            for img in images:
                # PIL Image -> base64 data URL
                buf = BytesIO()
                img.save(buf, format="JPEG")
                image_bytes = buf.getvalue()
                image_b64 = base64.b64encode(image_bytes).decode("utf-8")
                message_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_b64}"
                    }
                })

        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": message_content}
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=payload, timeout=self.timeout)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]


