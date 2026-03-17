"""
Half-life Predictor

Predicts elimination half-life properties including:
- Plasma half-life (t1/2)
- Elimination rate constant
- Clearance rate
"""

import numpy as np
from typing import Dict, Optional, Any
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Crippen, rdMolDescriptors

from .base_predictor import BaseADMETPredictor
from .descriptor_extractor import DescriptorExtractor
from .utils.model_loader import ModelLoader


class HalfLifePredictor(BaseADMETPredictor):
    """
    Predicts elimination half-life of molecules.
    
    Uses machine learning models or rule-based approaches to predict:
    - Plasma half-life (hours)
    - Elimination rate constant
    - Clearance classification
    """
    
    def __init__(self, model_path: Optional[str] = None, use_ml: bool = True):
        """
        Initialize half-life predictor.
        
        Args:
            model_path: Path to pre-trained model file
            use_ml: Whether to use ML model (True) or rule-based (False)
        """
        super().__init__(model_path, use_ml)
        self.descriptor_extractor = DescriptorExtractor(include_fingerprints=True)
        
        # Rule-based thresholds
        # Longer half-life: higher MW, higher LogP, lower clearance
        self.half_life_factors = {
            'short': {'mw': 300, 'logp': 2.0, 'clearance': 'high'},
            'medium': {'mw': 500, 'logp': 4.0, 'clearance': 'moderate'},
            'long': {'mw': 600, 'logp': 5.0, 'clearance': 'low'},
        }
    
    def _load_model(self):
        """Load pre-trained half-life model."""
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
        """Extract features for half-life prediction."""
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
            
            # If prediction is a half-life value
            if isinstance(prediction, (int, float, np.number)):
                half_life_hours = float(prediction)
                # Classify based on value
                if half_life_hours < 4:
                    classification = "Short (<4h)"
                elif half_life_hours < 24:
                    classification = "Medium (4-24h)"
                else:
                    classification = "Long (>24h)"
            else:
                half_life_hours = None
                classification = str(prediction)
            
            return {
                "prediction": {
                    "half_life_hours": half_life_hours,
                    "classification": classification
                },
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
        logp = Crippen.MolLogP(mol)
        hbd = Lipinski.NumHDonors(mol)
        hba = Lipinski.NumHAcceptors(mol)
        rot_bonds = Descriptors.NumRotatableBonds(mol)
        num_rings = rdMolDescriptors.CalcNumRings(mol)
        tpsa = Descriptors.TPSA(mol)
        
        # Calculate half-life score (higher = longer half-life)
        half_life_score = 0.0
        half_life_factors = []
        
        # Molecular weight factor (higher MW = longer half-life)
        if mw > self.half_life_factors['long']['mw']:
            half_life_score += 0.3
            half_life_factors.append("High MW favors longer half-life")
        elif mw > self.half_life_factors['medium']['mw']:
            half_life_score += 0.2
            half_life_factors.append("Moderate MW")
        elif mw < self.half_life_factors['short']['mw']:
            half_life_score -= 0.2
            half_life_factors.append("Low MW favors shorter half-life")
        
        # LogP factor (higher LogP = longer half-life, but very high may reduce)
        if 2.0 <= logp <= 5.0:
            half_life_score += 0.25
            half_life_factors.append("Optimal LogP for longer half-life")
        elif logp > 5.0:
            half_life_score += 0.1
            half_life_factors.append("Very high LogP (may reduce bioavailability)")
        elif logp < 1.0:
            half_life_score -= 0.2
            half_life_factors.append("Low LogP favors shorter half-life")
        
        # Ring count factor (more rings = longer half-life)
        if num_rings >= 3:
            half_life_score += 0.15
            half_life_factors.append("Multiple rings favor longer half-life")
        elif num_rings == 0:
            half_life_score -= 0.1
            half_life_factors.append("No rings may reduce half-life")
        
        # Rotatable bonds factor (fewer = longer half-life)
        if rot_bonds <= 5:
            half_life_score += 0.1
            half_life_factors.append("Low flexibility favors longer half-life")
        elif rot_bonds > 15:
            half_life_score -= 0.1
            half_life_factors.append("High flexibility may reduce half-life")
        
        # TPSA factor (lower TPSA = longer half-life)
        if tpsa < 60:
            half_life_score += 0.1
            half_life_factors.append("Low TPSA favors longer half-life")
        elif tpsa > 120:
            half_life_score -= 0.1
            half_life_factors.append("High TPSA may reduce half-life")
        
        # HBD/HBA factors (fewer = longer half-life)
        if hbd <= 3:
            half_life_score += 0.05
        if hba <= 6:
            half_life_score += 0.05
        
        # Normalize score
        half_life_score = max(0.0, min(1.0, half_life_score))
        
        # Estimate half-life in hours (simplified model)
        # Base half-life: 2-4 hours
        # Scale by score: 0.0 -> 2h, 1.0 -> 48h
        base_half_life = 2.0
        max_half_life = 48.0
        estimated_half_life = base_half_life + (max_half_life - base_half_life) * half_life_score
        
        # Classify half-life
        if estimated_half_life < 4:
            classification = "Short (<4h)"
            clearance = "High"
        elif estimated_half_life < 24:
            classification = "Medium (4-24h)"
            clearance = "Moderate"
        else:
            classification = "Long (>24h)"
            clearance = "Low"
        
        # Calculate elimination rate constant (k = ln(2) / t1/2)
        elimination_rate_constant = np.log(2) / estimated_half_life if estimated_half_life > 0 else 0.0
        
        return {
            "prediction": {
                "half_life_hours": float(estimated_half_life),
                "classification": classification,
                "elimination_rate_constant_per_hour": float(elimination_rate_constant),
                "clearance_classification": clearance,
                "half_life_score": float(half_life_score)
            },
            "confidence": 0.6,
            "details": {
                "half_life_factors": half_life_factors,
                "molecular_properties": {
                    "molecular_weight": float(mw),
                    "logp": float(logp),
                    "tpsa": float(tpsa),
                    "num_hbd": int(hbd),
                    "num_hba": int(hba),
                    "num_rotatable_bonds": int(rot_bonds),
                    "num_rings": int(num_rings)
                }
            }
        }


import os

