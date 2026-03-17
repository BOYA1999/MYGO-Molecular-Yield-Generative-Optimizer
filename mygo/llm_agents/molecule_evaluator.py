import json
import logging
from typing import Dict, Optional, List, Any
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen
from .base_llm import BaseLLM

logger = logging.getLogger(__name__)

class MoleculeEvaluator:
    def __init__(self, llm: BaseLLM):
        self.llm = llm
        # System prompt expects strict JSON output
        self.system_prompt = """You are an expert medicinal chemist. 
Evaluate molecules for drug discovery. 
You MUST respond with a valid JSON object containing the following keys:
{
    "drug_likeness_assessment": "string",
    "chemical_validity": "string",
    "strengths": ["string", "string"],
    "weaknesses": ["string", "string"],
    "optimization_suggestions": ["string", "string"]
}
Do not include markdown formatting (```json) in the response."""
    
    # Async API
    async def evaluate_molecule(
        self,
        smiles: str,
        docking_score: Optional[float] = None,
        properties: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return {"success": False, "error": "Invalid SMILES"}
        
        if properties is None:
            properties = self._compute_properties(mol)
        
        prompt = f"""Evaluate SMILES: {smiles}
Properties: {json.dumps(properties)}
Docking Score: {docking_score if docking_score else 'N/A'}
Provide structured JSON evaluation."""
        
        # Async generation with json_mode (if the backend supports it)
        response_text = await self.llm.generate(
            prompt=prompt,
            system_prompt=self.system_prompt,
            temperature=0.4,
            json_mode=True 
        )
        
        if not response_text:
            return {"success": False, "error": "LLM generation failed"}
            
        try:
            # Strip markdown fences defensively
            clean_text = response_text.replace("```json", "").replace("```", "").strip()
            evaluation_data = json.loads(clean_text)
            
            return {
                "success": True,
                "properties": properties,
                "drug_likeness_score": self._compute_drug_likeness_score(properties),
                "evaluation": evaluation_data,  # structured output
                "optimization_suggestions": evaluation_data.get("optimization_suggestions", [])
            }
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON: {response_text}")
            return {"success": False, "error": "JSON parsing failed"}

    # _compute_properties and _compute_drug_likeness_score are kept minimal here
    def _compute_properties(self, mol: Chem.Mol) -> Dict[str, float]:
        # ... (placeholder; extend as needed)
        return {
            "mw": Descriptors.MolWt(mol),
            "logp": Crippen.MolLogP(mol),
            "hbd": Descriptors.NumHDonors(mol),
            "hba": Descriptors.NumHAcceptors(mol),
            # SA score can be added if a scorer is available
        }
    
    # ... (additional helpers omitted)