"""
Claude 3.5 Sonnet Agent Implementation

Supports Anthropic Claude API.
"""

import os
import time
from typing import List, Dict, Optional
import requests
from .base_llm import BaseLLM


class ClaudeAgent(BaseLLM):
    """
    Claude 3.5 Sonnet agent using Anthropic API.
    """
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "claude-3-5-sonnet-20241022", **kwargs):
        """
        Initialize Claude agent.
        
        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            model_name: Model name (default: "claude-3-5-sonnet-20241022")
            **kwargs: Additional configuration
        """
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Anthropic API key is required. Set ANTHROPIC_API_KEY environment variable.")
        
        super().__init__(model_name=model_name, api_key=api_key, **kwargs)
        self.api_base = kwargs.get("api_base", "https://api.anthropic.com/v1")
        self.max_retries = kwargs.get("max_retries", 3)
        self.timeout = kwargs.get("timeout", 180)
    
    def _send_request(self, messages: List[Dict[str, str]], **kwargs) -> Optional[str]:
        """Send request to Anthropic API."""
        url = f"{self.api_base}/messages"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }
        
        # Convert messages format (separate system message)
        system_messages = [msg["content"] for msg in messages if msg["role"] == "system"]
        conversation_messages = [msg for msg in messages if msg["role"] != "system"]
        
        payload = {
            "model": self.model_name,
            "messages": conversation_messages,
            "max_tokens": kwargs.get("max_tokens", 4096),
        }
        
        if system_messages:
            payload["system"] = "\n".join(system_messages)
        
        if "temperature" in kwargs:
            payload["temperature"] = kwargs["temperature"]
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
                response.raise_for_status()
                
                data = response.json()
                content = data["content"][0]["text"]
                
                # Update token usage if available
                if "usage" in data:
                    usage = data["usage"]
                    self.total_tokens += usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
                
                return content
                
            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries - 1:
                    print(f"Request failed (attempt {attempt + 1}/{self.max_retries}), retrying...")
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    print(f"Request failed after {self.max_retries} attempts: {e}")
                    return None
        
        return None

