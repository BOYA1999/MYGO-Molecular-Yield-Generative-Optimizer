"""
GPT-4 Agent Implementation

Client for a chat-completions compatible API.
"""

import os
import time
from typing import List, Dict, Optional
import requests
from .base_llm import BaseLLM


class GPT4Agent(BaseLLM):
    """
    GPT-4 agent using a chat-completions API.
    """
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gpt-4", **kwargs):
        """
        Initialize GPT-4 agent.
        
        Args:
            api_key: API key (defaults to OPENAI_API_KEY env var)
            model_name: Model name (default: "gpt-4")
            **kwargs: Additional configuration
        """
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("API key is required. Set OPENAI_API_KEY environment variable.")
        
        super().__init__(model_name=model_name, api_key=api_key, **kwargs)
        self.api_base = kwargs.get("api_base", "https://api.openai.com/v1")
        self.max_retries = kwargs.get("max_retries", 3)
        self.timeout = kwargs.get("timeout", 180)
    
    def _send_request(self, messages: List[Dict[str, str]], **kwargs) -> Optional[str]:
        """Send request to chat-completions endpoint."""
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
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
                response.raise_for_status()
                
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                
                # Update token usage if available
                if "usage" in data:
                    usage = data["usage"]
                    self.total_tokens += usage.get("total_tokens", 0)
                
                return content
                
            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries - 1:
                    print(f"Request failed (attempt {attempt + 1}/{self.max_retries}), retrying...")
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    print(f"Request failed after {self.max_retries} attempts: {e}")
                    return None
        
        return None

