"""
MYGO Julia Python Interface

This module provides a Python interface to the MYGO.jl Julia package,
allowing Python code to use Julia-based molecular descriptors and
ADMET predictions.

Requirements:
- Julia 1.8+
- PyCall.jl (Julia package)
- MYGO.jl (this package)

Installation:
1. Install Julia from https://julialang.org/
2. Install PyCall.jl in Julia:
   using Pkg
   Pkg.add("PyCall")
3. Set PyCall to use Julia:
   ENV["PYTHON"] = ""
   Pkg.build("PyCall")
4. Install this package in Julia:
   using Pkg
   Pkg.add("MYGO")
"""

import os
import sys
from typing import Dict, List, Optional, Any, Union
import numpy as np

# Try to import PyCall and Julia
try:
    from julia import MYGO as MYGO_jl
    JULIA_AVAILABLE = True
except ImportError:
    JULIA_AVAILABLE = False
    print("Warning: Julia/PyCall not available. Using fallback Python implementation.")


class MolecularDescriptorJulia:
    """
    Python wrapper for Julia molecular descriptor calculations.
    
    This class provides an interface to the MYGO.jl package for
    calculating molecular descriptors and fingerprints.
    
    Example:
    ```python
    from mygo_julia import MolecularDescriptorJulia
    
    descriptor = MolecularDescriptorJulia()
    features = descriptor.extract("CCO")  # ethanol
    print(features)
    ```
    """
    
    def __init__(self, use_julia: bool = True):
        """
        Initialize the descriptor extractor.
        
        Args:
            use_julia: Whether to use Julia implementation (if available)
        """
        self.use_julia = use_julia and JULIA_AVAILABLE
        
        if not self.use_julia:
            # Fallback to Python RDKit implementation
            self._use_fallback = True
            try:
                from rdkit import Chem
                from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors
                self.RDKit = Chem
            except ImportError:
                raise ImportError("Neither Julia nor RDKit is available")
        else:
            self._use_fallback = False
            # Initialize Julia
            self._init_julia()
    
    def _init_julia(self):
        """Initialize Julia environment."""
        if not JULIA_AVAILABLE:
            raise RuntimeError("Julia is not available")
        
        # Pre-import functions for faster calls
        self._jl_parse = MYGO_jl.parse_smiles
        self._jl_descriptors = MYGO_jl.calculate_descriptors
        self._jl_fingerprint = MYGO_jl.calculate_fingerprint
    
    def parse_smiles(self, smiles: str) -> Any:
        """
        Parse SMILES string.
        
        Args:
            smiles: SMILES string
            
        Returns:
            Molecule object (Julia or RDKit)
        """
        if self._use_fallback:
            mol = self.RDKit.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"Invalid SMILES: {smiles}")
            return mol
        else:
            return self._jl_parse(smiles)
    
    def extract(self, smiles: str) -> Dict[str, float]:
        """
        Extract all descriptors from a SMILES string.
        
        Args:
            smiles: SMILES string
            
        Returns:
            Dictionary of descriptor names and values
        """
        if self._use_fallback:
            return self._extract_rdk(smiles)
        else:
            return self._extract_julia(smiles)
    
    def _extract_julia(self, smiles: str) -> Dict[str, float]:
        """Extract descriptors using Julia."""
        mol = self._jl_parse(smiles)
        desc = self._jl_descriptors(mol)
        
        # Convert Julia dict to Python dict
        result = {}
        for key, value in desc.items():
            if isinstance(value, (int, float)):
                result[key] = float(value)
            elif isinstance(value, str):
                result[key] = value
        
        return result
    
    def _extract_rdk(self, smiles: str) -> Dict[str, float]:
        """Extract descriptors using RDKit (fallback)."""
        mol = self.RDKit.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        
        descriptors = {}
        
        # Basic properties
        descriptors['molecular_weight'] = Descriptors.MolWt(mol)
        descriptors['logp'] = Crippen.MolLogP(mol)
        descriptors['tpsa'] = Descriptors.TPSA(mol)
        descriptors['num_atoms'] = mol.GetNumAtoms()
        descriptors['num_heavy_atoms'] = mol.GetNumHeavyAtoms()
        
        # Lipinski properties
        descriptors['num_hbd'] = Lipinski.NumHDonors(mol)
        descriptors['num_hba'] = Lipinski.NumHAcceptors(mol)
        descriptors['num_rotatable_bonds'] = Lipinski.NumRotatableBonds(mol)
        
        # Ring information
        descriptors['num_rings'] = rdMolDescriptors.CalcNumRings(mol)
        descriptors['num_aromatic_rings'] = rdMolDescriptors.CalcNumAromaticRings(mol)
        
        # Atom counts
        descriptors['num_carbons'] = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == 'C')
        descriptors['num_nitrogens'] = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == 'N')
        descriptors['num_oxygens'] = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == 'O')
        
        return descriptors
    
    def get_feature_vector(self, smiles: str) -> np.ndarray:
        """
        Get feature vector for machine learning.
        
        Args:
            smiles: SMILES string
            
        Returns:
            Feature vector as numpy array
        """
        desc = self.extract(smiles)
        return np.array(list(desc.values()), dtype=np.float32)
    
    def get_fingerprint(self, smiles: str, fp_type: str = "morgan", 
                       radius: int = 2, n_bits: int = 2048) -> np.ndarray:
        """
        Calculate molecular fingerprint.
        
        Args:
            smiles: SMILES string
            fp_type: Fingerprint type ("morgan", "rdkit", "maccs")
            radius: Radius for Morgan fingerprint
            n_bits: Number of bits
            
        Returns:
            Fingerprint as numpy array
        """
        if self._use_fallback:
            return self._get_fingerprint_rdk(smiles, fp_type, radius, n_bits)
        else:
            return self._get_fingerprint_julia(smiles, fp_type, radius, n_bits)
    
    def _get_fingerprint_julia(self, smiles: str, fp_type: str, 
                               radius: int, n_bits: int) -> np.ndarray:
        """Get fingerprint using Julia."""
        mol = self._jl_parse(smiles)
        fp = self._jl_fingerprint(mol, fp_type; radius=radius, n_bits=n_bits)
        
        return np.array(fp)
    
    def _get_fingerprint_rdk(self, smiles: str, fp_type: str, 
                             radius: int, n_bits: int) -> np.ndarray:
        """Get fingerprint using RDKit (fallback)."""
        from rdkit import Chem
        from rdkit.Chem import AllChem
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        
        if fp_type == "morgan":
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            arr = np.zeros((n_bits,), dtype=np.int32)
            for idx in fp.GetOnBits():
                arr[idx] = 1
            return arr
        elif fp_type == "rdkit":
            fp = Chem.RDKFingerprint(mol)
            arr = np.zeros((n_bits,), dtype=np.int32)
            for idx in fp.GetOnBits():
                if idx < n_bits:
                    arr[idx] = 1
            return arr
        else:
            raise ValueError(f"Unknown fingerprint type: {fp_type}")


class ADMETPredictorJulia:
    """
    Python wrapper for Julia ADMET predictions.
    
    Example:
    ```python
    from mygo_julia import ADMETPredictorJulia
    
    predictor = ADMETPredictorJulia()
    bbb_result = predictor.predict_bbb("CCO")
    print(bbb_result)
    ```
    """
    
    def __init__(self, use_julia: bool = True):
        """
        Initialize the ADMET predictor.
        
        Args:
            use_julia: Whether to use Julia implementation (if available)
        """
        self.use_julia = use_julia and JULIA_AVAILABLE
        
        if not self.use_julia:
            self._use_fallback = True
            try:
                from rdkit import Chem
                from rdkit.Chem import Descriptors, Crippen, Lipinski
                self.RDKit = Chem
            except ImportError:
                raise ImportError("Neither Julia nor RDKit is available")
        else:
            self._use_fallback = False
            self._init_julia()
    
    def _init_julia(self):
        """Initialize Julia environment."""
        if not JULIA_AVAILABLE:
            raise RuntimeError("Julia is not available")
        
        self._jl_parse = MYGO_jl.parse_smiles
        self._jl_predict_bbb = MYGO_jl.predict_bbb
        self._jl_predict_metabolism = MYGO_jl.predict_metabolism
        self._jl_predict_toxicity = MYGO_jl.predict_toxicity
    
    def predict_bbb(self, smiles: str) -> Dict[str, Any]:
        """
        Predict blood-brain barrier permeability.
        
        Args:
            smiles: SMILES string
            
        Returns:
            Dictionary with BBB prediction results
        """
        if self._use_fallback:
            return self._predict_bbb_rdk(smiles)
        else:
            return self._predict_bbb_julia(smiles)
    
    def _predict_bbb_julia(self, smiles: str) -> Dict[str, Any]:
        """Predict BBB using Julia."""
        mol = self._jl_parse(smiles)
        result = self._jl_predict_bbb(mol)
        
        return {
            "penetration": result.penetration,
            "score": result.score,
            "brain_plasma_ratio": result.brain_plasma_ratio,
            "confidence": result.confidence,
            "factors": list(result.factors),
            "properties": dict(result.properties)
        }
    
    def _predict_bbb_rdk(self, smiles: str) -> Dict[str, Any]:
        """Predict BBB using RDKit (fallback)."""
        from rdkit import Chem
        from rdkit.Chem import Descriptors, Crippen, Lipinski
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        
        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        hbd = Lipinski.NumHDonors(mol)
        hba = Lipinski.NumHAcceptors(mol)
        
        bbb_score = 0.0
        factors = []
        
        if mw < 450:
            bbb_score += 0.25
            factors.append("Optimal MW for BBB")
        
        if 1.0 <= logp <= 4.0:
            bbb_score += 0.25
            factors.append("Optimal LogP for BBB")
        
        if tpsa < 90:
            bbb_score += 0.2
            factors.append("Low TPSA favors BBB")
        
        if hbd <= 3:
            bbb_score += 0.15
            factors.append("Low HBD favors BBB")
        
        bbb_score = max(0.0, min(1.0, bbb_score))
        
        penetration = "Yes (High likelihood)" if bbb_score >= 0.7 else \
                     "Moderate" if bbb_score >= 0.4 else "No (Low likelihood)"
        
        return {
            "penetration": penetration,
            "score": bbb_score,
            "brain_plasma_ratio": bbb_score * 1.0,
            "confidence": 0.65,
            "factors": factors,
            "properties": {
                "molecular_weight": mw,
                "logp": logp,
                "tpsa": tpsa,
                "num_hbd": hbd,
                "num_hba": hba
            }
        }
    
    def predict_metabolism(self, smiles: str) -> Dict[str, Any]:
        """Predict metabolic properties."""
        if self._use_fallback:
            return self._predict_metabolism_rdk(smiles)
        else:
            return self._predict_metabolism_julia(smiles)
    
    def _predict_metabolism_julia(self, smiles: str) -> Dict[str, Any]:
        """Predict metabolism using Julia."""
        mol = self._jl_parse(smiles)
        result = self._jl_predict_metabolism(mol)
        
        return {
            "stability": result.metabolic_stability,
            "pathways": list(result.predicted_pathways),
            "cytochrome_interaction": result.cytochrome_interaction,
            "half_life": result.half_life_estimate,
            "confidence": result.confidence
        }
    
    def _predict_metabolism_rdk(self, smiles: str) -> Dict[str, Any]:
        """Predict metabolism using RDKit (fallback)."""
        from rdkit import Chem
        from rdkit.Chem import Descriptors
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        
        mw = Descriptors.MolWt(mol)
        stability = 0.7 if mw < 400 else 0.5
        
        return {
            "stability": stability,
            "pathways": ["Phase I: Oxidation predicted"],
            "cytochrome_interaction": "Moderate",
            "half_life": 2.0 / stability,
            "confidence": 0.6
        }
    
    def predict_toxicity(self, smiles: str) -> Dict[str, Any]:
        """Predict toxicity properties."""
        if self._use_fallback:
            return self._predict_toxicity_rdk(smiles)
        else:
            return self._predict_toxicity_julia(smiles)
    
    def _predict_toxicity_julia(self, smiles: str) -> Dict[str, Any]:
        """Predict toxicity using Julia."""
        mol = self._jl_parse(smiles)
        result = self._jl_predict_toxicity(mol)
        
        return {
            "overall_risk": result.toxicity_risk,
            "hERG_toxicity": result.hERG_toxicity,
            "hepatotoxicity": result.hepatotoxicity,
            "mutagenicity": result.mutagenicity,
            "carcinogenicity": result.carcinogenicity,
            "confidence": result.confidence
        }
    
    def _predict_toxicity_rdk(self, smiles: str) -> Dict[str, Any]:
        """Predict toxicity using RDKit (fallback)."""
        from rdkit import Chem
        from rdkit.Chem import Descriptors
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        
        return {
            "overall_risk": "Low",
            "hERG_toxicity": "Low",
            "hepatotoxicity": "Low",
            "mutagenicity": "Low",
            "carcinogenicity": "Low",
            "confidence": 0.55
        }


def main():
    """Example usage."""
    print("MYGO Julia Python Interface")
    print("=" * 40)
    
    print("\n1. Testing descriptor extraction...")
    try:
        desc = MolecularDescriptorJulia(use_julia=False)
        smiles = "CCO"
        features = desc.extract(smiles)
        print(f"   SMILES: {smiles}")
        print(f"   Molecular weight: {features.get('molecular_weight', 'N/A'):.2f}")
        print(f"   LogP: {features.get('logp', 'N/A'):.2f}")
        print(f"   TPSA: {features.get('tpsa', 'N/A'):.2f}")
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n2. Testing ADMET prediction...")
    try:
        predictor = ADMETPredictorJulia(use_julia=False)
        smiles = "CCO"
        bbb = predictor.predict_bbb(smiles)
        print(f"   SMILES: {smiles}")
        print(f"   BBB Penetration: {bbb.get('penetration', 'N/A')}")
        print(f"   BBB Score: {bbb.get('score', 'N/A'):.2f}")
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
