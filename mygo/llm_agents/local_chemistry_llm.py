"""
Local Chemistry Language Model for Molecular Yield Generative Optimizer (MYGO)

This module provides an optional local (on-prem) language model backend.

Notes:
- Local deployment typically requires a GPU and additional dependencies.
- API-backed agents and local agents share the same `BaseLLM` interface.
"""

import logging
import torch
from typing import List, Dict, Optional, Any
from llm_agents.base_llm import BaseLLM

logger = logging.getLogger(__name__)

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers library not available. Local LLM requires: pip install transformers torch")


class LocalChemistryLLM(BaseLLM):
    """
    Local deployment of chemistry-focused language model.
    
    This implementation uses domain-specific pre-trained models (e.g., ChemBERTa)
    or fine-tuned general models for chemistry tasks. It runs entirely on local
    hardware, ensuring data privacy but requiring significant computational resources.
    
    Resource Requirements:
    - GPU: 16GB+ VRAM (32GB+ recommended)
    - Model size: 1-7B parameters typically
    - Inference time: 0.5-2 seconds per request (vs 2-5s for API)
    """
    
    def __init__(self, model_name: str = "seyonec/ChemBERTa-zinc-base-v1", 
                 model_type: str = "classification",  # "classification" or "generation"
                 device: str = "cuda",
                 **kwargs):
        """
        Initialize local chemistry LLM.
        
        Args:
            model_name: HuggingFace model identifier or local path
            model_type: "classification" for property prediction, "generation" for text generation
            device: "cuda" or "cpu"
            **kwargs: Additional configuration
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers library required for local LLM. "
                "Install with: pip install transformers torch"
            )
        
        super().__init__(model_name=model_name, **kwargs)
        self.model_type = model_type
        self.device = device if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Loading local chemistry model: {model_name} on {self.device}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if model_type == "generation":
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None
                )
            else:
                # For classification/embedding tasks
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None
                )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            self.model.eval()
            logger.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(
                f"Local LLM initialization failed. "
                f"This requires significant computational resources. "
                f"Consider using API-based LLMs (GPT-4, Claude, DeepSeek) instead."
            ) from e
        
        # Chemistry knowledge base (rule-based fallback for simple queries)
        self.chemistry_kb = {
            "hydrogen_bond_donors": ["N", "O", "F"],
            "hydrogen_bond_acceptors": ["N", "O", "F"],
            "hydrophobic_atoms": ["C", "S"],
            "aromatic_atoms": ["C", "N"],
        }
    
    async def _send_request_async(self, messages: List[Dict[str, str]], **kwargs) -> Optional[str]:
        """
        Process request using local model.
        
        Note: This is a simplified implementation. For production use,
        consider implementing proper prompt engineering and task-specific
        fine-tuning for better performance.
        """
        prompt = messages[-1]["content"] if messages else ""
        
        # Simple rule-based responses for common queries (fast path)
        if self._is_simple_query(prompt):
            return self._rule_based_response(prompt)
        
        # Use model for complex queries
        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                if self.model_type == "generation":
                    outputs = self.model.generate(
                        **inputs,
                        max_length=kwargs.get("max_tokens", 256),
                        temperature=kwargs.get("temperature", 0.7),
                        do_sample=True
                    )
                    response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                else:
                    # Classification/embedding model - return feature-based response
                    outputs = self.model(**inputs)
                    # Convert to text response (simplified)
                    response = self._format_classification_response(outputs, prompt)
            
            return response
            
        except Exception as e:
            logger.error(f"Local model inference failed: {e}")
            # Fallback to rule-based
            return self._rule_based_response(prompt)
    
    def _is_simple_query(self, prompt: str) -> bool:
        """Check if query can be answered with rules"""
        simple_keywords = [
            "hydrogen bond", "hydrophobic", "aromatic",
            "atom type", "functional group"
        ]
        return any(keyword in prompt.lower() for keyword in simple_keywords)
    
    def _rule_based_response(self, prompt: str) -> str:
        """Rule-based response for simple queries"""
        prompt_lower = prompt.lower()
        
        if "hydrogen bond" in prompt_lower:
            return (
                "Recommended atom types for hydrogen bonding: N, O, F. "
                "These atoms can act as both donors and acceptors. "
                "Optimal distance: 2.5-3.5 Å."
            )
        elif "hydrophobic" in prompt_lower:
            return (
                "Hydrophobic regions favor: C, S atoms. "
                "Avoid polar atoms (N, O) in hydrophobic pockets."
            )
        else:
            return (
                "Based on standard medicinal chemistry principles: "
                "Consider molecular weight < 500 Da, LogP 0-5, "
                "HBD ≤ 5, HBA ≤ 10 for drug-likeness."
            )
    
    def _format_classification_response(self, outputs, prompt: str) -> str:
        """Format model outputs as text response"""
        # This is a placeholder - actual implementation depends on model architecture
        return "Local model analysis completed. (Classification model response)"
    
    def get_resource_usage(self) -> Dict[str, Any]:
        """Get current resource usage"""
        if self.device == "cuda":
            return {
                "device": self.device,
                "gpu_memory_allocated": torch.cuda.memory_allocated() / 1e9,  # GB
                "gpu_memory_reserved": torch.cuda.memory_reserved() / 1e9,  # GB
            }
        return {"device": self.device}


# Usage example and warnings
if __name__ == "__main__":
    print("""
    WARNING: Local LLM Deployment
    
    This module is designed for research groups with:
    1. High-end computational resources (A100/H100 GPUs, 32GB+ VRAM)
    2. Data privacy requirements that justify the computational cost
    3. Expertise in model deployment and optimization
    
    For most users, API-based LLMs (GPT-4, Claude, DeepSeek) provide:
    - Lower upfront costs (no GPU investment)
    - Better performance (larger, better-trained models)
    - Easier maintenance (no model updates/deployment)
    - Cost efficiency: API costs are ~0.1-1% of synthesis costs
    
    If you still want to use local LLM:
    1. Ensure you have sufficient GPU resources
    2. Install dependencies: pip install transformers torch
    3. Choose an appropriate model (ChemBERTa, MolBERT, etc.)
    4. Fine-tune on your specific tasks for best results
    """)

