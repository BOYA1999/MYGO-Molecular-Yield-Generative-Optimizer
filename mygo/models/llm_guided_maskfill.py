"""
LLM-guided Molecular Yield Generative Optimizer (MYGO) Model Wrapper

Integrates LLM guidance into the Molecular Yield Generative Optimizer (MYGO) diffusion model.
"""

import torch
from torch.nn import Module
from typing import Dict, Optional, List, Any
from .maskfill import PMAsymDenoiser

# Import LLM agents
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm_agents import PocketAnalyzer, GenerationAdvisor, MoleculeEvaluator


class LLMGuidedPMAsymDenoiser(Module):
    """
    Wrapper for PMAsymDenoiser with LLM guidance capabilities.
    
    This wrapper adds three levels of LLM guidance:
    1. Initialization guidance (pocket analysis)
    2. Intermediate evaluation (during generation)
    3. Post-generation optimization (molecule evaluation)
    """
    
    def __init__(
        self,
        base_model: PMAsymDenoiser,
        pocket_analyzer: Optional[PocketAnalyzer] = None,
        generation_advisor: Optional[GenerationAdvisor] = None,
        molecule_evaluator: Optional[MoleculeEvaluator] = None,
        use_llm_guidance: bool = True,
        guidance_frequency: int = 20,  # Evaluate every N steps
    ):
        """
        Initialize LLM-guided model.
        
        Args:
            base_model: Base PMAsymDenoiser model
            pocket_analyzer: PocketAnalyzer instance (optional)
            generation_advisor: GenerationAdvisor instance (optional)
            molecule_evaluator: MoleculeEvaluator instance (optional)
            use_llm_guidance: Whether to enable LLM guidance
            guidance_frequency: Frequency of intermediate evaluation (every N steps)
        """
        super().__init__()
        self.base_model = base_model
        self.pocket_analyzer = pocket_analyzer
        self.generation_advisor = generation_advisor
        self.molecule_evaluator = molecule_evaluator
        self.use_llm_guidance = use_llm_guidance
        self.guidance_frequency = guidance_frequency
        
        # Store guidance results
        self.pocket_guidance = None
        self.intermediate_evaluations = []
        self.final_evaluations = []
    
    def forward(self, batch, **kwargs):
        """
        Forward pass through the base model.
        
        Args:
            batch: Input batch
            **kwargs: Additional arguments
            
        Returns:
            Model outputs
        """
        # Just pass through to base model
        # LLM guidance is handled in the sampling loop
        return self.base_model(batch, **kwargs)
    
    def analyze_pocket(self, pocket_pdb_path: str) -> Optional[Dict[str, Any]]:
        """
        Analyze protein pocket and get initialization guidance.
        
        Args:
            pocket_pdb_path: Path to pocket PDB file
            
        Returns:
            Guidance dictionary or None
        """
        if not self.use_llm_guidance or not self.pocket_analyzer:
            return None
        
        try:
            with open(pocket_pdb_path, 'r') as f:
                pocket_content = f.read()
            
            result = self.pocket_analyzer.analyze_pocket(
                pocket_pdb_content=pocket_content,
                pocket_name=os.path.basename(pocket_pdb_path)
            )
            
            if result["success"]:
                self.pocket_guidance = result["guidance"]
                return result
        except Exception as e:
            print(f"Error analyzing pocket: {e}")
        
        return None
    
    def evaluate_intermediate(
        self,
        batch,
        outputs,
        step: int,
        total_steps: int
    ) -> Optional[Dict[str, Any]]:
        """
        Evaluate intermediate structure during generation.
        
        Args:
            batch: Current batch
            outputs: Model outputs
            step: Current step
            total_steps: Total steps
            
        Returns:
            Evaluation dictionary or None
        """
        if not self.use_llm_guidance or not self.generation_advisor:
            return None
        
        # Only evaluate at specified frequency
        if step % self.guidance_frequency != 0:
            return None
        
        try:
            # Extract SMILES if possible (requires reconstruction)
            # This is a simplified version - full implementation would decode the molecule
            smiles = None  # TODO: Decode from outputs
            
            # Extract atom types
            atom_types = None
            if 'pred_node' in outputs:
                # Convert predicted node types to atom type names
                # This is simplified - actual implementation would map indices to atom types
                atom_types = ["C", "N", "O"]  # Placeholder
            
            result = self.generation_advisor.evaluate_intermediate(
                smiles=smiles,
                atom_types=atom_types,
                step=step,
                total_steps=total_steps
            )
            
            if result["success"]:
                self.intermediate_evaluations.append(result)
            
            return result
        except Exception as e:
            print(f"Error evaluating intermediate: {e}")
        
        return None
    
    def evaluate_molecule(
        self,
        smiles: str,
        docking_score: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Evaluate final generated molecule.
        
        Args:
            smiles: SMILES string of generated molecule
            docking_score: Optional docking score
            
        Returns:
            Evaluation dictionary or None
        """
        if not self.use_llm_guidance or not self.molecule_evaluator:
            return None
        
        try:
            result = self.molecule_evaluator.evaluate_molecule(
                smiles=smiles,
                docking_score=docking_score
            )
            
            if result["success"]:
                self.final_evaluations.append(result)
            
            return result
        except Exception as e:
            print(f"Error evaluating molecule: {e}")
        
        return None
    
    def get_guidance_summary(self) -> Dict[str, Any]:
        """
        Get summary of all LLM guidance results.
        
        Returns:
            Dictionary with guidance summary
        """
        return {
            "pocket_guidance": self.pocket_guidance,
            "intermediate_evaluations_count": len(self.intermediate_evaluations),
            "final_evaluations_count": len(self.final_evaluations),
            "intermediate_evaluations": self.intermediate_evaluations[-5:] if self.intermediate_evaluations else [],  # Last 5
            "final_evaluations": self.final_evaluations[-5:] if self.final_evaluations else [],  # Last 5
        }

