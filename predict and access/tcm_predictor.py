"""
TCM (Teratogenicity, Carcinogenicity, Mutagenicity) Predictor

Predicts TCM properties including:
- Teratogenicity (birth defects)
- Carcinogenicity (cancer risk)
- Mutagenicity (genetic mutations)
"""

import numpy as np
from typing import Dict, Optional, Any
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

from .base_predictor import BaseADMETPredictor
from .descriptor_extractor import DescriptorExtractor
from .utils.model_loader import ModelLoader


class TCMPredictor(BaseADMETPredictor):
    """
    Predicts TCM (Teratogenicity, Carcinogenicity, Mutagenicity) properties.
    
    Uses machine learning models or rule-based approaches to predict:
    - Teratogenicity risk
    - Carcinogenicity risk
    - Mutagenicity risk
    """
    
    def __init__(self, model_path: Optional[str] = None, use_ml: bool = True):
        """
        Initialize TCM predictor.
        
        Args:
            model_path: Path to pre-trained model file
            use_ml: Whether to use ML model (True) or rule-based (False)
        """
        super().__init__(model_path, use_ml)
        self.descriptor_extractor = DescriptorExtractor(include_fingerprints=True)
        
        # TCM structural alerts (simplified, based on known toxicophores)
        self.tcm_patterns = {
            'teratogenic': [
                '[N+](=O)[O-]',  # Nitro groups
                '[C;H2;H3][N+](=O)',  # Nitroso groups
                '[C;H2;H3]C(=O)O',  # Carboxylic acids (some)
                '[Cl,Br,I][c]',  # Halogenated aromatics
            ],
            'carcinogenic': [
                '[N+](=O)[O-]',  # Nitro groups
                '[C;H2;H3][N+](=O)',  # Nitroso groups
                '[c]1[c][c][c][c][c]1',  # Polycyclic aromatics
                '[C;H2;H3]C(=O)N',  # Amides (some)
                '[S;D2](=O)(=O)',  # Sulfonyl groups
            ],
            'mutagenic': [
                '[N+](=O)[O-]',  # Nitro groups
                '[C;H2;H3][N+](=O)',  # Nitroso groups
                '[N;H2]',  # Primary amines (some)
                '[C;H2;H3]C(=O)N',  # Amides (some)
                '[c]1[c][c][c][c][c]1',  # Aromatic rings (some)
            ]
        }
    
    def _load_model(self):
        """Load pre-trained TCM model."""
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
        """Extract features for TCM prediction."""
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
        Predict using rule-based approach with structural alerts.
        
        Args:
            mol: RDKit molecule object
            
        Returns:
            Prediction results dictionary
        """
        # Calculate molecular properties
        mw = Descriptors.MolWt(mol)
        num_rings = rdMolDescriptors.CalcNumRings(mol)
        num_aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
        
        # Teratogenicity prediction
        teratogenic_score = 0.0
        teratogenic_alerts = []
        
        for pattern_smarts in self.tcm_patterns['teratogenic']:
            pattern = Chem.MolFromSmarts(pattern_smarts)
            if pattern and mol.HasSubstructMatch(pattern):
                matches = mol.GetSubstructMatches(pattern)
                teratogenic_score += 0.4
                teratogenic_alerts.append(f"Found teratogenic pattern: {pattern_smarts} ({len(matches)} matches)")
        
        # High MW may increase teratogenicity risk
        if mw > 400:
            teratogenic_score += 0.1
            teratogenic_alerts.append("High molecular weight")
        
        teratogenic_score = min(1.0, teratogenic_score)
        teratogenicity = "High" if teratogenic_score >= 0.6 else "Medium" if teratogenic_score >= 0.3 else "Low"
        
        # Carcinogenicity prediction
        carcinogenic_score = 0.0
        carcinogenic_alerts = []
        
        for pattern_smarts in self.tcm_patterns['carcinogenic']:
            pattern = Chem.MolFromSmarts(pattern_smarts)
            if pattern and mol.HasSubstructMatch(pattern):
                matches = mol.GetSubstructMatches(pattern)
                carcinogenic_score += 0.35
                carcinogenic_alerts.append(f"Found carcinogenic pattern: {pattern_smarts} ({len(matches)} matches)")
        
        # Multiple aromatic rings increase carcinogenicity risk
        if num_aromatic_rings >= 3:
            carcinogenic_score += 0.3
            carcinogenic_alerts.append("Multiple aromatic rings (potential PAH-like structure)")
        
        # High number of rings
        if num_rings >= 4:
            carcinogenic_score += 0.2
            carcinogenic_alerts.append("High ring count")
        
        carcinogenic_score = min(1.0, carcinogenic_score)
        carcinogenicity = "High" if carcinogenic_score >= 0.6 else "Medium" if carcinogenic_score >= 0.3 else "Low"
        
        # Mutagenicity prediction
        mutagenic_score = 0.0
        mutagenic_alerts = []
        
        for pattern_smarts in self.tcm_patterns['mutagenic']:
            pattern = Chem.MolFromSmarts(pattern_smarts)
            if pattern and mol.HasSubstructMatch(pattern):
                matches = mol.GetSubstructMatches(pattern)
                mutagenic_score += 0.4
                mutagenic_alerts.append(f"Found mutagenic pattern: {pattern_smarts} ({len(matches)} matches)")
        
        # Aromatic rings may increase mutagenicity
        if num_aromatic_rings >= 2:
            mutagenic_score += 0.2
            mutagenic_alerts.append("Multiple aromatic rings")
        
        mutagenic_score = min(1.0, mutagenic_score)
        mutagenicity = "High" if mutagenic_score >= 0.6 else "Medium" if mutagenic_score >= 0.3 else "Low"
        
        # Overall TCM risk
        overall_tcm_score = max(teratogenic_score, carcinogenic_score, mutagenic_score)
        overall_tcm_risk = "High" if overall_tcm_score >= 0.6 else "Medium" if overall_tcm_score >= 0.3 else "Low"
        
        return {
            "prediction": {
                "overall_tcm_risk": overall_tcm_risk,
                "overall_tcm_score": float(overall_tcm_score),
                "teratogenicity": teratogenicity,
                "teratogenicity_score": float(teratogenic_score),
                "carcinogenicity": carcinogenicity,
                "carcinogenicity_score": float(carcinogenic_score),
                "mutagenicity": mutagenicity,
                "mutagenicity_score": float(mutagenic_score)
            },
            "confidence": 0.6,
            "details": {
                "teratogenic_alerts": teratogenic_alerts if teratogenic_alerts else ["No teratogenic alerts detected"],
                "carcinogenic_alerts": carcinogenic_alerts if carcinogenic_alerts else ["No carcinogenic alerts detected"],
                "mutagenic_alerts": mutagenic_alerts if mutagenic_alerts else ["No mutagenic alerts detected"],
                "molecular_properties": {
                    "molecular_weight": float(mw),
                    "num_rings": int(num_rings),
                    "num_aromatic_rings": int(num_aromatic_rings)
                }
            }
        }


import os

