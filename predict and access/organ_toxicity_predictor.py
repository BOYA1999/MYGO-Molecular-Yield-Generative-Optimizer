"""
Organ Toxicity Predictor

Predicts organ-specific toxicity including:
- Hepatotoxicity (liver)
- Nephrotoxicity (kidney)
- Cardiotoxicity (heart)
- Neurotoxicity (nervous system)
"""

import numpy as np
from typing import Dict, Optional, Any, List
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Crippen, rdMolDescriptors

from .base_predictor import BaseADMETPredictor
from .descriptor_extractor import DescriptorExtractor
from .utils.model_loader import ModelLoader


class OrganToxicityPredictor(BaseADMETPredictor):
    """
    Predicts organ-specific toxicity.
    
    Uses machine learning models or rule-based approaches to predict:
    - Hepatotoxicity (liver toxicity)
    - Nephrotoxicity (kidney toxicity)
    - Cardiotoxicity (heart toxicity)
    - Neurotoxicity (nervous system toxicity)
    """
    
    def __init__(self, model_path: Optional[str] = None, use_ml: bool = True):
        """
        Initialize organ toxicity predictor.
        
        Args:
            model_path: Path to pre-trained model file
            use_ml: Whether to use ML model (True) or rule-based (False)
        """
        super().__init__(model_path, use_ml)
        self.descriptor_extractor = DescriptorExtractor(include_fingerprints=True)
        
        # Toxic structural alerts (simplified)
        self.toxic_patterns = {
            'hepatotoxic': [
                '[C;H2;H3][N+](=O)[O-]',  # Nitro groups
                '[C;H2;H3][N+](=O)',  # Nitroso groups
                '[Cl,Br,I][c]',  # Halogenated aromatics
                '[S;D2](=O)(=O)',  # Sulfonyl groups
            ],
            'nephrotoxic': [
                '[C;H2;H3][N+](=O)[O-]',  # Nitro groups
                '[C;H2;H3]C(=O)O',  # Carboxylic acids
                '[C;H2;H3][S;D2](=O)(=O)',  # Sulfonamides
            ],
            'cardiotoxic': [
                '[N+](=O)[O-]',  # Nitro groups
                '[c]1[c][c][c][c][c]1',  # Aromatic rings (simplified)
                '[C;H2;H3][N+](=O)',  # Nitroso groups
            ],
            'neurotoxic': [
                '[C;H2;H3][N+](=O)[O-]',  # Nitro groups
                '[C;H2;H3][S;D2](=O)(=O)',  # Sulfonyl groups
                '[C;H2;H3]C(=O)N',  # Amides (some)
            ]
        }
    
    def _load_model(self):
        """Load pre-trained organ toxicity model."""
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
        """Extract features for organ toxicity prediction."""
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
        logp = Crippen.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        num_rings = rdMolDescriptors.CalcNumRings(mol)
        num_aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
        
        # Predict toxicity for each organ
        organ_toxicities = {}
        
        # Hepatotoxicity prediction
        hepatotoxic_score = 0.0
        hepatotoxic_alerts = []
        
        for pattern_smarts in self.toxic_patterns['hepatotoxic']:
            pattern = Chem.MolFromSmarts(pattern_smarts)
            if pattern and mol.HasSubstructMatch(pattern):
                matches = mol.GetSubstructMatches(pattern)
                hepatotoxic_score += 0.3
                hepatotoxic_alerts.append(f"Found hepatotoxic pattern: {pattern_smarts} ({len(matches)} matches)")
        
        # High MW increases hepatotoxicity risk
        if mw > 500:
            hepatotoxic_score += 0.2
            hepatotoxic_alerts.append("High molecular weight")
        
        # High LogP increases hepatotoxicity risk
        if logp > 5.0:
            hepatotoxic_score += 0.2
            hepatotoxic_alerts.append("High LogP")
        
        hepatotoxic_score = min(1.0, hepatotoxic_score)
        hepatotoxicity = "High" if hepatotoxic_score >= 0.6 else "Medium" if hepatotoxic_score >= 0.3 else "Low"
        
        # Nephrotoxicity prediction
        nephrotoxic_score = 0.0
        nephrotoxic_alerts = []
        
        for pattern_smarts in self.toxic_patterns['nephrotoxic']:
            pattern = Chem.MolFromSmarts(pattern_smarts)
            if pattern and mol.HasSubstructMatch(pattern):
                matches = mol.GetSubstructMatches(pattern)
                nephrotoxic_score += 0.3
                nephrotoxic_alerts.append(f"Found nephrotoxic pattern: {pattern_smarts} ({len(matches)} matches)")
        
        # High TPSA may indicate nephrotoxicity
        if tpsa > 120:
            nephrotoxic_score += 0.15
            nephrotoxic_alerts.append("High TPSA")
        
        nephrotoxic_score = min(1.0, nephrotoxic_score)
        nephrotoxicity = "High" if nephrotoxic_score >= 0.6 else "Medium" if nephrotoxic_score >= 0.3 else "Low"
        
        # Cardiotoxicity prediction
        cardiotoxic_score = 0.0
        cardiotoxic_alerts = []
        
        for pattern_smarts in self.toxic_patterns['cardiotoxic']:
            pattern = Chem.MolFromSmarts(pattern_smarts)
            if pattern and mol.HasSubstructMatch(pattern):
                matches = mol.GetSubstructMatches(pattern)
                cardiotoxic_score += 0.3
                cardiotoxic_alerts.append(f"Found cardiotoxic pattern: {pattern_smarts} ({len(matches)} matches)")
        
        # Multiple aromatic rings may indicate cardiotoxicity
        if num_aromatic_rings >= 3:
            cardiotoxic_score += 0.2
            cardiotoxic_alerts.append("Multiple aromatic rings")
        
        cardiotoxic_score = min(1.0, cardiotoxic_score)
        cardiotoxicity = "High" if cardiotoxic_score >= 0.6 else "Medium" if cardiotoxic_score >= 0.3 else "Low"
        
        # Neurotoxicity prediction
        neurotoxic_score = 0.0
        neurotoxic_alerts = []
        
        for pattern_smarts in self.toxic_patterns['neurotoxic']:
            pattern = Chem.MolFromSmarts(pattern_smarts)
            if pattern and mol.HasSubstructMatch(pattern):
                matches = mol.GetSubstructMatches(pattern)
                neurotoxic_score += 0.3
                neurotoxic_alerts.append(f"Found neurotoxic pattern: {pattern_smarts} ({len(matches)} matches)")
        
        # High LogP may increase neurotoxicity
        if logp > 4.0:
            neurotoxic_score += 0.15
            neurotoxic_alerts.append("High LogP")
        
        neurotoxic_score = min(1.0, neurotoxic_score)
        neurotoxicity = "High" if neurotoxic_score >= 0.6 else "Medium" if neurotoxic_score >= 0.3 else "Low"
        
        # Overall toxicity score (maximum of all organ toxicities)
        overall_toxicity_score = max(hepatotoxic_score, nephrotoxic_score, cardiotoxic_score, neurotoxic_score)
        overall_toxicity = "High" if overall_toxicity_score >= 0.6 else "Medium" if overall_toxicity_score >= 0.3 else "Low"
        
        return {
            "prediction": {
                "overall_toxicity": overall_toxicity,
                "overall_toxicity_score": float(overall_toxicity_score),
                "hepatotoxicity": hepatotoxicity,
                "hepatotoxicity_score": float(hepatotoxic_score),
                "nephrotoxicity": nephrotoxicity,
                "nephrotoxicity_score": float(nephrotoxic_score),
                "cardiotoxicity": cardiotoxicity,
                "cardiotoxicity_score": float(cardiotoxic_score),
                "neurotoxicity": neurotoxicity,
                "neurotoxicity_score": float(neurotoxic_score)
            },
            "confidence": 0.6,
            "details": {
                "hepatotoxic_alerts": hepatotoxic_alerts if hepatotoxic_alerts else ["No hepatotoxic alerts detected"],
                "nephrotoxic_alerts": nephrotoxic_alerts if nephrotoxic_alerts else ["No nephrotoxic alerts detected"],
                "cardiotoxic_alerts": cardiotoxic_alerts if cardiotoxic_alerts else ["No cardiotoxic alerts detected"],
                "neurotoxic_alerts": neurotoxic_alerts if neurotoxic_alerts else ["No neurotoxic alerts detected"],
                "molecular_properties": {
                    "molecular_weight": float(mw),
                    "logp": float(logp),
                    "tpsa": float(tpsa),
                    "num_rings": int(num_rings),
                    "num_aromatic_rings": int(num_aromatic_rings)
                }
            }
        }


import os

