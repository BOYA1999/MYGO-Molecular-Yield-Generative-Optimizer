import os
import logging
import aiohttp  # async HTTP client
import asyncio
from typing import List, Dict, Optional, Any
# from .base_llm import BaseLLM

logger = logging.getLogger(__name__)

class DeepSeekAgent:  # lightweight async client (can be adapted to BaseLLM)
    def __init__(self, api_key: Optional[str] = None, model_name: str = "deepseek-chat", **kwargs):
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("DeepSeek API key is required.")
        self.model_name = model_name
        # Base URL may or may not include a /v1 prefix depending on the provider setup.
        self.api_base = kwargs.get("api_base", "https://api.deepseek.com")
        self.max_retries = kwargs.get("max_retries", 3)
        self.timeout = kwargs.get("timeout", 180)

    async def _send_request_async(self, messages: List[Dict[str, str]], **kwargs) -> str:
        url = f"{self.api_base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model_name,
            "messages": messages,
            **kwargs
        }

        async with aiohttp.ClientSession() as session:
            for attempt in range(self.max_retries):
                try:
                    async with session.post(url, json=payload, headers=headers, timeout=self.timeout) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            logger.error(f"API Error {response.status}: {error_text}")
                            response.raise_for_status()
                        
                        data = await response.json()
                        return data["choices"][0]["message"]["content"]
                        
                except Exception as e:
                    logger.warning(f"Request failed (attempt {attempt + 1}): {e}")
                    if attempt == self.max_retries - 1:
                        raise e  # raise on final attempt
                    await asyncio.sleep(2 ** attempt)