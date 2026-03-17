"""
Metabolism Predictor

Predicts metabolic properties of molecules including:
- Metabolic stability
- Metabolic pathways
- CYP450 interactions
- Metabolic clearance
"""

import os
import warnings
import numpy as np
from typing import Dict, Optional, Any, List
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors

from .base_predictor import BaseADMETPredictor
from .descriptor_extractor import DescriptorExtractor
from .utils.model_loader import ModelLoader


class MetabolismPredictor(BaseADMETPredictor):
    """
    Predicts metabolic properties of molecules.
    
    Uses machine learning models or rule-based approaches to predict:
    - Metabolic stability (high/medium/low)
    - CYP450 substrate likelihood
    - Metabolic clearance rate
    - Primary metabolic sites
    """
    
    def __init__(self, model_path: Optional[str] = None, use_ml: bool = True):
        """
        Initialize metabolism predictor.
        
        Args:
            model_path: Path to pre-trained model file
            use_ml: Whether to use ML model (True) or rule-based (False)
        """
        super().__init__(model_path, use_ml)
        self.descriptor_extractor = DescriptorExtractor(include_fingerprints=True)
        
        # Rule-based thresholds (based on literature)
        self.stability_thresholds = {
            'high': {'mw': 500, 'logp': 3.0, 'hbd': 3, 'hba': 6},
            'medium': {'mw': 600, 'logp': 4.0, 'hbd': 5, 'hba': 8},
        }
    
    def _load_model(self):
        """Load pre-trained metabolism model."""
        if self.model_path and os.path.exists(self.model_path):
            try:
                self.model = ModelLoader.load_model(self.model_path)
                self.is_loaded = True
            except Exception as e:
                warnings.warn(f"Failed to load model from {self.model_path}: {e}")
                self.model = None
                self.is_loaded = False
        else:
            self.model = None
            self.is_loaded = False
    
    def _extract_features(self, mol: Chem.Mol) -> np.ndarray:
        """Extract features for metabolism prediction."""
        return self.descriptor_extractor.extract_feature_vector(mol)
    
    def _predict_ml(self, features: np.ndarray) -> Dict[str, Any]:
        """
        Predict using machine learning model.
        
        Args:
            features: Feature vector
            
        Returns:
            Prediction results dictionary
        """
        if self.model is None:
            # Fallback to rule-based if model not available
            return self._predict_rule_based(Chem.MolFromSmiles('C'))  # Dummy mol
        
        try:
            # Reshape features if needed
            if len(features.shape) == 1:
                features = features.reshape(1, -1)
            
            # Predict
            if hasattr(self.model, 'predict_proba'):
                # Classification model
                proba = self.model.predict_proba(features)[0]
                prediction = self.model.predict(features)[0]
                confidence = float(np.max(proba))
            elif hasattr(self.model, 'predict'):
                # Regression or classification
                prediction = self.model.predict(features)[0]
                confidence = 0.8  # Default confidence
            else:
                raise ValueError("Model does not have predict method")
            
            return {
                "prediction": float(prediction) if isinstance(prediction, (int, float, np.number)) else str(prediction),
                "confidence": float(confidence),
                "details": {
                    "model_type": type(self.model).__name__,
                    "features_used": len(features[0])
                }
            }
        except Exception as e:
            return {
                "prediction": None,
                "confidence": 0.0,
                "error": str(e)
            }
    
    def _predict_rule_based(self, mol: Chem.Mol) -> Dict[str, Any]:
        """
        Predict using rule-based approach.
        
        Args:
            mol: RDKit molecule object
            
        Returns:
            Prediction results dictionary
        """
        # Calculate molecular properties
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.Crippen.MolLogP(mol)
        hbd = Lipinski.NumHDonors(mol)
        hba = Lipinski.NumHAcceptors(mol)
        num_rings = rdMolDescriptors.CalcNumRings(mol)
        num_aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
        
        # Predict metabolic stability
        stability_score = 0
        stability_factors = []
        
        # Molecular weight factor
        if mw < self.stability_thresholds['high']['mw']:
            stability_score += 0.2
            stability_factors.append("Low MW favors stability")
        elif mw < self.stability_thresholds['medium']['mw']:
            stability_score += 0.1
            stability_factors.append("Moderate MW")
        else:
            stability_factors.append("High MW may reduce stability")
        
        # LogP factor
        if 0 < logp < self.stability_thresholds['high']['logp']:
            stability_score += 0.2
            stability_factors.append("Optimal LogP")
        elif logp < self.stability_thresholds['medium']['logp']:
            stability_score += 0.1
            stability_factors.append("Moderate LogP")
        else:
            stability_factors.append("High LogP may increase metabolism")
        
        # HBD factor
        if hbd <= self.stability_thresholds['high']['hbd']:
            stability_score += 0.2
            stability_factors.append("Low HBD count")
        elif hbd <= self.stability_thresholds['medium']['hbd']:
            stability_score += 0.1
            stability_factors.append("Moderate HBD count")
        else:
            stability_factors.append("High HBD may increase metabolism")
        
        # HBA factor
        if hba <= self.stability_thresholds['high']['hba']:
            stability_score += 0.2
            stability_factors.append("Low HBA count")
        elif hba <= self.stability_thresholds['medium']['hba']:
            stability_score += 0.1
            stability_factors.append("Moderate HBA count")
        else:
            stability_factors.append("High HBA may increase metabolism")
        
        # Aromatic rings factor (more aromatic rings may increase metabolism)
        if num_aromatic_rings <= 2:
            stability_score += 0.1
            stability_factors.append("Moderate aromatic content")
        elif num_aromatic_rings > 3:
            stability_score -= 0.1
            stability_factors.append("High aromatic content may increase CYP450 metabolism")
        
        # Determine stability category
        if stability_score >= 0.7:
            stability = "High"
        elif stability_score >= 0.4:
            stability = "Medium"
        else:
            stability = "Low"
        
        # Predict CYP450 substrate likelihood
        cyp450_score = 0.0
        cyp450_factors = []
        
        # Aromatic rings increase CYP450 interaction
        if num_aromatic_rings >= 2:
            cyp450_score += 0.3
            cyp450_factors.append("Multiple aromatic rings")
        
        # Nitrogen-containing heterocycles
        num_nitrogens = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == 'N')
        if num_nitrogens >= 2:
            cyp450_score += 0.2
            cyp450_factors.append("Multiple nitrogen atoms")
        
        # High LogP increases CYP450 interaction
        if logp > 3.0:
            cyp450_score += 0.2
            cyp450_factors.append("High LogP")
        
        # Molecular weight factor
        if 200 < mw < 500:
            cyp450_score += 0.3
            cyp450_factors.append("Optimal MW range")
        
        cyp450_likelihood = "High" if cyp450_score >= 0.6 else "Medium" if cyp450_score >= 0.3 else "Low"
        
        # Estimate clearance (simplified)
        clearance_rate = "Fast" if stability_score < 0.4 else "Moderate" if stability_score < 0.7 else "Slow"
        
        return {
            "prediction": {
                "metabolic_stability": stability,
                "stability_score": float(stability_score),
                "cyp450_substrate_likelihood": cyp450_likelihood,
                "cyp450_score": float(cyp450_score),
                "estimated_clearance": clearance_rate
            },
            "confidence": 0.6,  # Rule-based confidence
            "details": {
                "stability_factors": stability_factors,
                "cyp450_factors": cyp450_factors,
                "molecular_properties": {
                    "molecular_weight": float(mw),
                    "logp": float(logp),
                    "num_hbd": int(hbd),
                    "num_hba": int(hba),
                    "num_aromatic_rings": int(num_aromatic_rings)
                }
            }
        }
    
    def predict_metabolic_sites(self, mol: Chem.Mol) -> List[Dict[str, Any]]:
        """
        Predict potential metabolic sites in the molecule.
        
        Args:
            mol: RDKit molecule object
            
        Returns:
            List of dictionaries with atom indices and metabolic reactions
        """
        metabolic_sites = []
        
        # Common metabolic reactions
        # 1. Aromatic hydroxylation
        aromatic_pattern = Chem.MolFromSmarts('[c;H]')
        aromatic_matches = mol.GetSubstructMatches(aromatic_pattern)
        for match in aromatic_matches:
            metabolic_sites.append({
                "atom_idx": match[0],
                "reaction": "Aromatic hydroxylation",
                "likelihood": "High"
            })
        
        # 2. Aliphatic hydroxylation
        aliphatic_pattern = Chem.MolFromSmarts('[C;H2;!$(C=*)]')
        aliphatic_matches = mol.GetSubstructMatches(aliphatic_pattern)
        for match in aliphatic_matches[:3]:  # Limit to top 3
            metabolic_sites.append({
                "atom_idx": match[0],
                "reaction": "Aliphatic hydroxylation",
                "likelihood": "Medium"
            })
        
        # 3. N-dealkylation
        n_alkyl_pattern = Chem.MolFromSmarts('[N;!H0;$(N-C)]')
        n_alkyl_matches = mol.GetSubstructMatches(n_alkyl_pattern)
        for match in n_alkyl_matches:
            metabolic_sites.append({
                "atom_idx": match[0],
                "reaction": "N-dealkylation",
                "likelihood": "Medium"
            })
        
        return metabolic_sites
