import json
import re

class PocketAnalyzer:
    def __init__(self, llm: BaseLLM):
        self.llm = llm
        self.system_prompt = """Analyze protein pockets. 
Output JSON format:
{
    "key_residues": ["res1", "res2"],
    "pocket_properties": "string",
    "pharmacophore_features": ["feature1", "feature2"],
    "initialization_guidance": {
        "mw_range": [min, max],
        "logp_range": [min, max],
        "description": "string"
    }
}"""
    
    def _sanitize_pdb(self, pdb_content: str) -> str:
        """Keep key lines only to reduce tokens."""
        lines = [line for line in pdb_content.split('\n') 
                 if line.startswith(('ATOM', 'HETATM'))]
        return '\n'.join(lines[:150])  # hard cap; replace with smarter truncation if needed

    async def analyze_pocket(self, pocket_pdb_content: str, pocket_name: str = "") -> Dict[str, Any]:
        
        compact_pdb = self._sanitize_pdb(pocket_pdb_content)
        
        prompt = f"""Analyze Pocket: {pocket_name}
Structure Snippet:
{compact_pdb}

Provide initialization guidance in JSON."""
        
        response_text = await self.llm.generate(
            prompt=prompt,
            system_prompt=self.system_prompt,
            temperature=0.3,
            max_tokens=1000,
            json_mode=True
        )
        
        # NOTE: JSON parsing/validation is expected here.
        # Keeping the response in structured JSON avoids additional regex post-processing.