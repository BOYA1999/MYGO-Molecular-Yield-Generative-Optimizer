import json
from typing import Dict, Optional, List, Any
from rdkit import Chem
# import numpy as np  # keep optional heavy deps at module level if needed

class GenerationAdvisor:
    def __init__(self, llm: BaseLLM):
        self.llm = llm
        self.system_prompt = """You are a computational chemist analyzing intermediate structures.
Return ONLY a JSON object:
{
    "validity_assessment": "string",
    "issues": ["string"],
    "binding_potential": "string",
    "recommendations": ["string", "string"]
}"""
    
    async def evaluate_intermediate(
        self,
        smiles: Optional[str] = None,
        step: int = 0,
        total_steps: int = 100,
        **kwargs
    ) -> Dict[str, Any]:
        
        # Keep prompt compact to reduce overhead and improve stability
        description = f"Step {step}/{total_steps}. "
        if smiles:
            description += f"SMILES: {smiles}"
        
        # ... (optionally include geometry/constraints; keep instructions brief)
        
        prompt = f"""Data: {description}
Analyze the intermediate structure. JSON format."""

        # Lower max_tokens for latency; lower temperature for format stability
        response_text = await self.llm.generate(
            prompt=prompt,
            system_prompt=self.system_prompt,
            temperature=0.2, 
            max_tokens=300,
            json_mode=True
        )
        
        if not response_text:
            return {"success": False}
            
        try:
            clean_text = response_text.replace("```json", "").replace("```", "").strip()
            feedback_data = json.loads(clean_text)
            return {
                "success": True,
                "step": step,
                "feedback": feedback_data,
                "recommendations": feedback_data.get("recommendations", [])
            }
        except Exception:
            return {"success": False, "raw_response": response_text}