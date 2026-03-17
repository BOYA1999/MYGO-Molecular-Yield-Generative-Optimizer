"""
Plasma Exposure Predictor

Predicts plasma exposure properties including:
- Cmax (maximum plasma concentration)
- AUC (area under the curve)
- Plasma protein binding
- Bioavailability
"""

import numpy as np
from typing import Dict, Optional, Any
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Crippen

from .base_predictor import BaseADMETPredictor
from .descriptor_extractor import DescriptorExtractor
from .utils.model_loader import ModelLoader


class PlasmaExposurePredictor(BaseADMETPredictor):
    """
    Predicts plasma exposure properties of molecules.
    
    Uses machine learning models or rule-based approaches to predict:
    - Maximum plasma concentration (Cmax)
    - Area under the curve (AUC)
    - Plasma protein binding percentage
    - Oral bioavailability
    """
    
    def __init__(self, model_path: Optional[str] = None, use_ml: bool = True):
        """
        Initialize plasma exposure predictor.
        
        Args:
            model_path: Path to pre-trained model file
            use_ml: Whether to use ML model (True) or rule-based (False)
        """
        super().__init__(model_path, use_ml)
        self.descriptor_extractor = DescriptorExtractor(include_fingerprints=True)
        
        # Rule-based thresholds
        self.bioavailability_thresholds = {
            'high': {'mw': 500, 'logp': 3.5, 'hbd': 5, 'hba': 10, 'rot_bonds': 10},
            'medium': {'mw': 600, 'logp': 5.0, 'hbd': 7, 'hba': 12, 'rot_bonds': 15},
        }
    
    def _load_model(self):
        """Load pre-trained plasma exposure model."""
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
        """Extract features for plasma exposure prediction."""
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
        tpsa = Descriptors.TPSA(mol)
        
        # Predict bioavailability
        bioav_score = 0.0
        bioav_factors = []
        
        # Molecular weight factor
        if mw < self.bioavailability_thresholds['high']['mw']:
            bioav_score += 0.2
            bioav_factors.append("Optimal MW for absorption")
        elif mw < self.bioavailability_thresholds['medium']['mw']:
            bioav_score += 0.1
            bioav_factors.append("Moderate MW")
        else:
            bioav_factors.append("High MW may reduce absorption")
        
        # LogP factor (optimal range 1-3)
        if 1.0 <= logp <= 3.5:
            bioav_score += 0.25
            bioav_factors.append("Optimal LogP for absorption")
        elif 0 <= logp <= 5.0:
            bioav_score += 0.15
            bioav_factors.append("Moderate LogP")
        else:
            bioav_factors.append("Suboptimal LogP")
        
        # HBD factor
        if hbd <= self.bioavailability_thresholds['high']['hbd']:
            bioav_score += 0.2
            bioav_factors.append("Low HBD favors absorption")
        elif hbd <= self.bioavailability_thresholds['medium']['hbd']:
            bioav_score += 0.1
            bioav_factors.append("Moderate HBD")
        else:
            bioav_factors.append("High HBD may reduce absorption")
        
        # HBA factor
        if hba <= self.bioavailability_thresholds['high']['hba']:
            bioav_score += 0.2
            bioav_factors.append("Low HBA favors absorption")
        elif hba <= self.bioavailability_thresholds['medium']['hba']:
            bioav_score += 0.1
            bioav_factors.append("Moderate HBA")
        else:
            bioav_factors.append("High HBA may reduce absorption")
        
        # Rotatable bonds factor
        if rot_bonds <= self.bioavailability_thresholds['high']['rot_bonds']:
            bioav_score += 0.15
            bioav_factors.append("Low flexibility favors absorption")
        else:
            bioav_factors.append("High flexibility may affect absorption")
        
        # Determine bioavailability category
        if bioav_score >= 0.8:
            bioavailability = "High (>70%)"
        elif bioav_score >= 0.5:
            bioavailability = "Medium (30-70%)"
        else:
            bioavailability = "Low (<30%)"
        
        # Predict plasma protein binding (simplified model)
        # Higher LogP and aromatic content increase binding
        ppb_score = 0.0
        if logp > 3.0:
            ppb_score += 0.3
        if logp > 4.0:
            ppb_score += 0.2
        
        num_aromatic_rings = Descriptors.NumAromaticRings(mol)
        if num_aromatic_rings >= 2:
            ppb_score += 0.3
        if num_aromatic_rings >= 3:
            ppb_score += 0.2
        
        # Estimate PPB percentage
        if ppb_score >= 0.7:
            ppb = "High (>90%)"
            ppb_percentage = 92.0
        elif ppb_score >= 0.4:
            ppb = "Medium (70-90%)"
            ppb_percentage = 80.0
        else:
            ppb = "Low (<70%)"
            ppb_percentage = 50.0
        
        # Estimate Cmax (simplified, in ng/mL)
        # Based on bioavailability and molecular properties
        base_cmax = 100.0  # Base value
        cmax_multiplier = bioav_score * 2.0  # Scale by bioavailability
        estimated_cmax = base_cmax * cmax_multiplier
        
        # Estimate AUC (simplified, in ng·h/mL)
        # AUC is related to clearance and bioavailability
        clearance_factor = 1.0 - (bioav_score * 0.5)  # Lower bioavailability = higher clearance
        estimated_auc = estimated_cmax / clearance_factor * 24.0  # Rough estimate
        
        return {
            "prediction": {
                "bioavailability": bioavailability,
                "bioavailability_score": float(bioav_score),
                "plasma_protein_binding": ppb,
                "plasma_protein_binding_percentage": float(ppb_percentage),
                "estimated_cmax_ng_ml": float(estimated_cmax),
                "estimated_auc_ng_h_ml": float(estimated_auc)
            },
            "confidence": 0.6,
            "details": {
                "bioavailability_factors": bioav_factors,
                "molecular_properties": {
                    "molecular_weight": float(mw),
                    "logp": float(logp),
                    "tpsa": float(tpsa),
                    "num_hbd": int(hbd),
                    "num_hba": int(hba),
                    "num_rotatable_bonds": int(rot_bonds),
                    "num_aromatic_rings": int(num_aromatic_rings)
                }
            }
        }


import os

