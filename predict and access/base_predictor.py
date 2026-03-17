"""
Base ADMET Predictor

Provides base class for all ADMET prediction modules.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Any, List, Union
import numpy as np
from rdkit import Chem


class BaseADMETPredictor(ABC):
    """
    Base class for all ADMET predictors.
    
    All predictors should inherit from this class and implement
    the abstract methods.
    """
    
    def __init__(self, model_path: Optional[str] = None, use_ml: bool = True):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to pre-trained model file (optional)
            use_ml: Whether to use machine learning model (True) or rule-based (False)
        """
        self.model_path = model_path
        self.use_ml = use_ml
        self.model = None
        self.is_loaded = False
        
        if model_path and use_ml:
            self._load_model()
    
    @abstractmethod
    def _load_model(self):
        """Load the pre-trained model from file."""
        pass
    
    @abstractmethod
    def _extract_features(self, mol: Chem.Mol) -> np.ndarray:
        """
        Extract features/descriptors from molecule.
        
        Args:
            mol: RDKit molecule object
            
        Returns:
            Feature vector as numpy array
        """
        pass
    
    @abstractmethod
    def _predict_ml(self, features: np.ndarray) -> Dict[str, Any]:
        """
        Make prediction using machine learning model.
        
        Args:
            features: Feature vector
            
        Returns:
            Dictionary with prediction results
        """
        pass
    
    @abstractmethod
    def _predict_rule_based(self, mol: Chem.Mol) -> Dict[str, Any]:
        """
        Make prediction using rule-based approach.
        
        Args:
            mol: RDKit molecule object
            
        Returns:
            Dictionary with prediction results
        """
        pass
    
    def predict(self, mol: Union[Chem.Mol, str]) -> Dict[str, Any]:
        """
        Predict ADMET property for a molecule.
        
        Args:
            mol: RDKit molecule object or SMILES string
            
        Returns:
            Dictionary with prediction results including:
            - prediction: Main prediction value
            - confidence: Confidence score (0-1)
            - details: Additional details
            - method: Method used ('ml' or 'rule_based')
        """
        # Convert SMILES to molecule if needed
        if isinstance(mol, str):
            mol = Chem.MolFromSmiles(mol)
            if mol is None:
                return {
                    "success": False,
                    "error": "Invalid SMILES string",
                    "prediction": None
                }
        
        if mol is None:
            return {
                "success": False,
                "error": "Invalid molecule",
                "prediction": None
            }
        
        try:
            if self.use_ml and self.model is not None:
                features = self._extract_features(mol)
                result = self._predict_ml(features)
                result["method"] = "ml"
            else:
                result = self._predict_rule_based(mol)
                result["method"] = "rule_based"
            
            result["success"] = True
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "prediction": None
            }
    
    def predict_batch(self, mols: List[Union[Chem.Mol, str]]) -> List[Dict[str, Any]]:
        """
        Predict for a batch of molecules.
        
        Args:
            mols: List of RDKit molecule objects or SMILES strings
            
        Returns:
            List of prediction result dictionaries
        """
        results = []
        for mol in mols:
            results.append(self.predict(mol))
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_path": self.model_path,
            "use_ml": self.use_ml,
            "is_loaded": self.is_loaded,
            "model_type": type(self.model).__name__ if self.model else None
        }
