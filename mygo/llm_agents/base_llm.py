import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any, Union

# Configure logger
logger = logging.getLogger(__name__)

class BaseLLM(ABC):
    """
    Async Base class for LLM interfaces.
    
    All LLM implementations should inherit from this class and implement
    the _send_request_async method.
    """
    
    def __init__(self, model_name: str, api_key: Optional[str] = None, **kwargs):
        """
        Initialize the LLM agent.
        
        Args:
            model_name: Name of the LLM model
            api_key: API key for the LLM service
            **kwargs: Additional configuration (e.g., cache_size)
        """
        self.model_name = model_name
        self.api_key = api_key
        self.config = kwargs
        self.context: List[Dict[str, str]] = []
        
        # Simple cache
        self.cache: Dict[str, str] = {}
        self.max_cache_size = kwargs.get("max_cache_size", 1000)
        
        # Usage statistics
        self.call_count = 0
        self.total_tokens = 0
        
    @abstractmethod
    async def _send_request_async(self, messages: List[Dict[str, str]], **kwargs) -> Optional[str]:
        """
        Abstract method to send an async request to the LLM API.
        
        Args:
            messages: List of message dictionaries
            **kwargs: Additional parameters (temperature, json_mode, etc.)
        """
        pass
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        use_cache: bool = True,
        json_mode: bool = False,  # normalized JSON-mode switch
        **kwargs
    ) -> Optional[str]:
        """
        Generate a response from the LLM asynchronously.
        
        Args:
            prompt: User prompt
            system_prompt: System prompt (optional)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            use_cache: Whether to use cached responses
            json_mode: Whether to force JSON output
            **kwargs: Additional parameters
            
        Returns:
            Generated response text, or None if failed
        """
        # Create cache key (including system prompt and json_mode)
        cache_key = f"{self.model_name}_{prompt}_{system_prompt}_{temperature}_{json_mode}"
        
        # Check cache
        if use_cache and cache_key in self.cache:
            logger.debug(f"Cache hit for prompt: {prompt[:30]}...")
            return self.cache[cache_key]
        
        # Build messages
        messages = []
        
        # Conversation context (caller should manage length)
        if self.context:
            messages.extend(self.context)
            
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
            
        messages.append({"role": "user", "content": prompt})
        
        # Prepare kwargs
        request_kwargs = {
            "temperature": temperature,
            "json_mode": json_mode,
            **kwargs
        }
        if max_tokens:
            request_kwargs["max_tokens"] = max_tokens
        
        # Send request
        try:
            self.call_count += 1
            response = await self._send_request_async(messages, **request_kwargs)
            
            if response:
                # Cache successful responses with size limit check
                if use_cache:
                    if len(self.cache) >= self.max_cache_size:
                        # FIFO eviction (switch to LRU if needed)
                        self.cache.pop(next(iter(self.cache)))
                    self.cache[cache_key] = response
                    
            return response
            
        except Exception as e:
            logger.error(f"Error generating response from {self.model_name}: {e}", exc_info=True)
            return None
    
    def reset_context(self):
        """Clear the conversation context."""
        self.context = []
        logger.info("Context reset.")
    
    def add_to_context(self, role: str, content: str):
        """Add a message to the conversation context."""
        self.context.append({"role": role, "content": content})
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about LLM usage."""
        return {
            "model_name": self.model_name,
            "call_count": self.call_count,
            "total_tokens": self.total_tokens,
            "cache_size": len(self.cache),
        }
    
    def clear_cache(self):
        """Clear the response cache."""
        self.cache.clear()
        logger.info("Cache cleared.")