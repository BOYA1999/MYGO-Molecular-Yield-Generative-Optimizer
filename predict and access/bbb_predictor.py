"""
Blood-Brain Barrier (BBB) Permeability Predictor

Predicts BBB permeability properties including:
- BBB penetration (Yes/No)
- Permeability score
- Brain-to-plasma ratio
"""

import numpy as np
from typing import Dict, Optional, Any
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Crippen

from .base_predictor import BaseADMETPredictor
from .descriptor_extractor import DescriptorExtractor
from .utils.model_loader import ModelLoader


class BBBPredictor(BaseADMETPredictor):
    """
    Predicts blood-brain barrier permeability.
    
    Uses machine learning models or rule-based approaches to predict:
    - BBB penetration likelihood
    - Permeability score
    - Brain-to-plasma ratio
    """
    
    def __init__(self, model_path: Optional[str] = None, use_ml: bool = True):
        """
        Initialize BBB predictor.
        
        Args:
            model_path: Path to pre-trained model file
            use_ml: Whether to use ML model (True) or rule-based (False)
        """
        super().__init__(model_path, use_ml)
        self.descriptor_extractor = DescriptorExtractor(include_fingerprints=True)
        
        # Rule-based thresholds (based on literature)
        # BBB+ compounds typically: MW < 450, LogP 1-4, TPSA < 90, HBD < 3
        self.bbb_thresholds = {
            'favorable': {'mw': 450, 'logp_min': 1.0, 'logp_max': 4.0, 'tpsa': 90, 'hbd': 3},
            'moderate': {'mw': 500, 'logp_min': 0.5, 'logp_max': 5.0, 'tpsa': 120, 'hbd': 5},
        }
    
    def _load_model(self):
        """Load pre-trained BBB model."""
        if self.model_path and os.path.exists(self.model_path):
            try:
                self.model = ModelLoader.load_model(self.model_path)
                self.is_loaded = True
            except Exception as e:
                import warnings
                warnings.warn(f"Failed to load model from {self.model_path}: {e}")
                self.model = None
                self.is_loaded = False
        else:
            self.model = None
            self.is_loaded = False
    
    def _extract_features(self, mol: Chem.Mol) -> np.ndarray:
        """Extract features for BBB prediction."""
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
            return self._predict_rule_based(Chem.MolFromSmiles('C'))
        
        try:
            if len(features.shape) == 1:
                features = features.reshape(1, -1)
            
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(features)[0]
                prediction = self.model.predict(features)[0]
                confidence = float(np.max(proba))
            elif hasattr(self.model, 'predict'):
                prediction = self.model.predict(features)[0]
                confidence = 0.8
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
        Predict using rule-based approach (based on known BBB rules).
        
        Args:
            mol: RDKit molecule object
            
        Returns:
            Prediction results dictionary
        """
        # Calculate molecular properties
        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        hbd = Lipinski.NumHDonors(mol)
        hba = Lipinski.NumHAcceptors(mol)
        rot_bonds = Descriptors.NumRotatableBonds(mol)
        
        # Calculate BBB permeability score
        bbb_score = 0.0
        bbb_factors = []
        
        # Molecular weight factor (lower is better for BBB)
        if mw < self.bbb_thresholds['favorable']['mw']:
            bbb_score += 0.25
            bbb_factors.append("Optimal MW for BBB penetration")
        elif mw < self.bbb_thresholds['moderate']['mw']:
            bbb_score += 0.15
            bbb_factors.append("Moderate MW")
        else:
            bbb_score -= 0.2
            bbb_factors.append("High MW reduces BBB penetration")
        
        # LogP factor (optimal range 1-4)
        logp_min = self.bbb_thresholds['favorable']['logp_min']
        logp_max = self.bbb_thresholds['favorable']['logp_max']
        if logp_min <= logp <= logp_max:
            bbb_score += 0.25
            bbb_factors.append("Optimal LogP for BBB penetration")
        elif self.bbb_thresholds['moderate']['logp_min'] <= logp <= self.bbb_thresholds['moderate']['logp_max']:
            bbb_score += 0.15
            bbb_factors.append("Moderate LogP")
        elif logp < 0:
            bbb_score -= 0.2
            bbb_factors.append("Very low LogP (too polar)")
        else:
            bbb_score -= 0.1
            bbb_factors.append("High LogP may reduce BBB penetration")
        
        # TPSA factor (lower is better)
        if tpsa < self.bbb_thresholds['favorable']['tpsa']:
            bbb_score += 0.2
            bbb_factors.append("Low TPSA favors BBB penetration")
        elif tpsa < self.bbb_thresholds['moderate']['tpsa']:
            bbb_score += 0.1
            bbb_factors.append("Moderate TPSA")
        else:
            bbb_score -= 0.2
            bbb_factors.append("High TPSA reduces BBB penetration")
        
        # HBD factor (lower is better)
        if hbd <= self.bbb_thresholds['favorable']['hbd']:
            bbb_score += 0.15
            bbb_factors.append("Low HBD count favors BBB penetration")
        elif hbd <= self.bbb_thresholds['moderate']['hbd']:
            bbb_score += 0.05
            bbb_factors.append("Moderate HBD count")
        else:
            bbb_score -= 0.15
            bbb_factors.append("High HBD count reduces BBB penetration")
        
        # Rotatable bonds factor (lower is better)
        if rot_bonds <= 10:
            bbb_score += 0.1
            bbb_factors.append("Low flexibility")
        elif rot_bonds > 15:
            bbb_score -= 0.1
            bbb_factors.append("High flexibility may reduce BBB penetration")
        
        # Normalize score to 0-1 range
        bbb_score = max(0.0, min(1.0, bbb_score))
        
        # Determine BBB penetration
        if bbb_score >= 0.7:
            bbb_penetration = "Yes (High likelihood)"
            brain_plasma_ratio = 0.8 + (bbb_score - 0.7) * 0.7  # 0.8-1.5
        elif bbb_score >= 0.4:
            bbb_penetration = "Moderate"
            brain_plasma_ratio = 0.3 + (bbb_score - 0.4) * 1.67  # 0.3-0.8
        else:
            bbb_penetration = "No (Low likelihood)"
            brain_plasma_ratio = bbb_score * 0.75  # 0-0.3
        
        return {
            "prediction": {
                "bbb_penetration": bbb_penetration,
                "permeability_score": float(bbb_score),
                "brain_to_plasma_ratio": float(brain_plasma_ratio),
                "prediction_binary": "Yes" if bbb_score >= 0.5 else "No"
            },
            "confidence": 0.65,
            "details": {
                "bbb_factors": bbb_factors,
                "molecular_properties": {
                    "molecular_weight": float(mw),
                    "logp": float(logp),
                    "tpsa": float(tpsa),
                    "num_hbd": int(hbd),
                    "num_hba": int(hba),
                    "num_rotatable_bonds": int(rot_bonds)
                },
                "rule_applied": "Modified Lipinski/CNS rules"
            }
        }


import os

