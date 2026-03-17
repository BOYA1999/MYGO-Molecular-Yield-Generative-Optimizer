"""
Ensemble ADMET Predictor

Combines all ADMET predictors to provide comprehensive evaluation.
"""

from typing import Dict, List, Optional, Any, Union
from rdkit import Chem
import pandas as pd
import numpy as np

from .base_predictor import BaseADMETPredictor
from .metabolism_predictor import MetabolismPredictor
from .plasma_exposure_predictor import PlasmaExposurePredictor
from .bbb_predictor import BBBPredictor
from .organ_toxicity_predictor import OrganToxicityPredictor
from .tcm_predictor import TCMPredictor
from .half_life_predictor import HalfLifePredictor


class EnsembleADMETPredictor:
    """
    Ensemble predictor that combines all ADMET prediction modules.
    
    Provides comprehensive ADMET evaluation for molecules.
    """
    
    def __init__(
        self,
        model_dir: Optional[str] = None,
        use_ml: bool = True,
        predictors_config: Optional[Dict[str, Dict]] = None
    ):
        """
        Initialize ensemble predictor.
        
        Args:
            model_dir: Directory containing pre-trained model files
            use_ml: Whether to use ML models (True) or rule-based (False)
            predictors_config: Configuration for individual predictors
        """
        self.model_dir = model_dir
        self.use_ml = use_ml
        self.predictors_config = predictors_config or {}
        
        # Initialize all predictors
        self.metabolism_predictor = MetabolismPredictor(
            model_path=self._get_model_path('metabolism'),
            use_ml=use_ml
        )
        
        self.plasma_exposure_predictor = PlasmaExposurePredictor(
            model_path=self._get_model_path('plasma_exposure'),
            use_ml=use_ml
        )
        
        self.bbb_predictor = BBBPredictor(
            model_path=self._get_model_path('bbb'),
            use_ml=use_ml
        )
        
        self.organ_toxicity_predictor = OrganToxicityPredictor(
            model_path=self._get_model_path('organ_toxicity'),
            use_ml=use_ml
        )
        
        self.tcm_predictor = TCMPredictor(
            model_path=self._get_model_path('tcm'),
            use_ml=use_ml
        )
        
        self.half_life_predictor = HalfLifePredictor(
            model_path=self._get_model_path('half_life'),
            use_ml=use_ml
        )
        
        self.predictors = {
            'metabolism': self.metabolism_predictor,
            'plasma_exposure': self.plasma_exposure_predictor,
            'bbb': self.bbb_predictor,
            'organ_toxicity': self.organ_toxicity_predictor,
            'tcm': self.tcm_predictor,
            'half_life': self.half_life_predictor,
        }
    
    def _get_model_path(self, predictor_name: str) -> Optional[str]:
        """Get model path for a predictor."""
        if self.model_dir:
            import os
            model_path = os.path.join(self.model_dir, f"{predictor_name}_model.pkl")
            if os.path.exists(model_path):
                return model_path
        return None
    
    def predict_all(self, mol: Union[Chem.Mol, str]) -> Dict[str, Any]:
        """
        Run all ADMET predictions for a molecule.
        
        Args:
            mol: RDKit molecule object or SMILES string
            
        Returns:
            Dictionary with all prediction results
        """
        # Convert SMILES to molecule if needed
        if isinstance(mol, str):
            mol = Chem.MolFromSmiles(mol)
            if mol is None:
                return {
                    "success": False,
                    "error": "Invalid SMILES string",
                    "predictions": {}
                }
        
        if mol is None:
            return {
                "success": False,
                "error": "Invalid molecule",
                "predictions": {}
            }
        
        # Get SMILES for reference
        smiles = Chem.MolToSmiles(mol)
        
        # Run all predictions
        results = {
            "smiles": smiles,
            "success": True,
            "predictions": {}
        }
        
        # Metabolism prediction
        try:
            metabolism_result = self.metabolism_predictor.predict(mol)
            results["predictions"]["metabolism"] = metabolism_result
        except Exception as e:
            results["predictions"]["metabolism"] = {
                "success": False,
                "error": str(e)
            }
        
        # Plasma exposure prediction
        try:
            plasma_result = self.plasma_exposure_predictor.predict(mol)
            results["predictions"]["plasma_exposure"] = plasma_result
        except Exception as e:
            results["predictions"]["plasma_exposure"] = {
                "success": False,
                "error": str(e)
            }
        
        # BBB prediction
        try:
            bbb_result = self.bbb_predictor.predict(mol)
            results["predictions"]["bbb"] = bbb_result
        except Exception as e:
            results["predictions"]["bbb"] = {
                "success": False,
                "error": str(e)
            }
        
        # Organ toxicity prediction
        try:
            organ_toxicity_result = self.organ_toxicity_predictor.predict(mol)
            results["predictions"]["organ_toxicity"] = organ_toxicity_result
        except Exception as e:
            results["predictions"]["organ_toxicity"] = {
                "success": False,
                "error": str(e)
            }
        
        # TCM prediction
        try:
            tcm_result = self.tcm_predictor.predict(mol)
            results["predictions"]["tcm"] = tcm_result
        except Exception as e:
            results["predictions"]["tcm"] = {
                "success": False,
                "error": str(e)
            }
        
        # Half-life prediction
        try:
            half_life_result = self.half_life_predictor.predict(mol)
            results["predictions"]["half_life"] = half_life_result
        except Exception as e:
            results["predictions"]["half_life"] = {
                "success": False,
                "error": str(e)
            }
        
        # Calculate overall ADMET score
        results["overall_admet_score"] = self._calculate_overall_score(results["predictions"])
        
        return results
    
    def predict_batch(self, mols: List[Union[Chem.Mol, str]]) -> List[Dict[str, Any]]:
        """
        Run predictions for a batch of molecules.
        
        Args:
            mols: List of RDKit molecule objects or SMILES strings
            
        Returns:
            List of prediction result dictionaries
        """
        results = []
        for mol in mols:
            results.append(self.predict_all(mol))
        return results
    
    def _calculate_overall_score(self, predictions: Dict[str, Any]) -> float:
        """
        Calculate overall ADMET score from all predictions.
        
        Args:
            predictions: Dictionary with all prediction results
            
        Returns:
            Overall ADMET score (0-1, higher is better)
        """
        score_components = []
        
        # Metabolism score (higher stability = better)
        if 'metabolism' in predictions and predictions['metabolism'].get('success'):
            metab_pred = predictions['metabolism'].get('prediction', {})
            if isinstance(metab_pred, dict):
                stability = metab_pred.get('metabolic_stability', '')
                if stability == 'High':
                    score_components.append(1.0)
                elif stability == 'Medium':
                    score_components.append(0.6)
                else:
                    score_components.append(0.3)
        
        # Plasma exposure score (higher bioavailability = better)
        if 'plasma_exposure' in predictions and predictions['plasma_exposure'].get('success'):
            plasma_pred = predictions['plasma_exposure'].get('prediction', {})
            if isinstance(plasma_pred, dict):
                bioav_score = plasma_pred.get('bioavailability_score', 0)
                score_components.append(bioav_score)
        
        # BBB score (context-dependent, assume moderate is good)
        if 'bbb' in predictions and predictions['bbb'].get('success'):
            bbb_pred = predictions['bbb'].get('prediction', {})
            if isinstance(bbb_pred, dict):
                bbb_score = bbb_pred.get('permeability_score', 0.5)
                score_components.append(0.7)  # Neutral for BBB (context-dependent)
        
        # Toxicity scores (lower = better)
        if 'organ_toxicity' in predictions and predictions['organ_toxicity'].get('success'):
            tox_pred = predictions['organ_toxicity'].get('prediction', {})
            if isinstance(tox_pred, dict):
                overall_tox = tox_pred.get('overall_toxicity_score', 0.5)
                score_components.append(1.0 - overall_tox)  # Invert (lower toxicity = higher score)
        
        if 'tcm' in predictions and predictions['tcm'].get('success'):
            tcm_pred = predictions['tcm'].get('prediction', {})
            if isinstance(tcm_pred, dict):
                tcm_score = tcm_pred.get('overall_tcm_score', 0.5)
                score_components.append(1.0 - tcm_score)  # Invert
        
        # Half-life score (moderate is often best)
        if 'half_life' in predictions and predictions['half_life'].get('success'):
            hl_pred = predictions['half_life'].get('prediction', {})
            if isinstance(hl_pred, dict):
                hl_hours = hl_pred.get('half_life_hours', 12)
                # Optimal range: 4-24 hours
                if 4 <= hl_hours <= 24:
                    score_components.append(1.0)
                elif 2 <= hl_hours < 4 or 24 < hl_hours <= 48:
                    score_components.append(0.7)
                else:
                    score_components.append(0.4)
        
        # Calculate average score
        if score_components:
            overall_score = np.mean(score_components)
        else:
            overall_score = 0.5  # Default neutral score
        
        return float(overall_score)
    
    def get_summary(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get summary of predictions.
        
        Args:
            predictions: Prediction results dictionary
            
        Returns:
            Summary dictionary
        """
        summary = {
            "total_predictions": len(predictions),
            "successful_predictions": sum(1 for p in predictions.values() if p.get('success', False)),
            "failed_predictions": sum(1 for p in predictions.values() if not p.get('success', False)),
        }
        
        # Extract key predictions
        key_predictions = {}
        
        if 'metabolism' in predictions and predictions['metabolism'].get('success'):
            pred = predictions['metabolism'].get('prediction', {})
            if isinstance(pred, dict):
                key_predictions['metabolic_stability'] = pred.get('metabolic_stability', 'Unknown')
        
        if 'plasma_exposure' in predictions and predictions['plasma_exposure'].get('success'):
            pred = predictions['plasma_exposure'].get('prediction', {})
            if isinstance(pred, dict):
                key_predictions['bioavailability'] = pred.get('bioavailability', 'Unknown')
        
        if 'bbb' in predictions and predictions['bbb'].get('success'):
            pred = predictions['bbb'].get('prediction', {})
            if isinstance(pred, dict):
                key_predictions['bbb_penetration'] = pred.get('bbb_penetration', 'Unknown')
        
        if 'organ_toxicity' in predictions and predictions['organ_toxicity'].get('success'):
            pred = predictions['organ_toxicity'].get('prediction', {})
            if isinstance(pred, dict):
                key_predictions['overall_toxicity'] = pred.get('overall_toxicity', 'Unknown')
        
        if 'tcm' in predictions and predictions['tcm'].get('success'):
            pred = predictions['tcm'].get('prediction', {})
            if isinstance(pred, dict):
                key_predictions['overall_tcm_risk'] = pred.get('overall_tcm_risk', 'Unknown')
        
        if 'half_life' in predictions and predictions['half_life'].get('success'):
            pred = predictions['half_life'].get('prediction', {})
            if isinstance(pred, dict):
                key_predictions['half_life'] = pred.get('classification', 'Unknown')
        
        summary['key_predictions'] = key_predictions
        
        return summary

