"""
Model Loading Utilities

Provides utilities for loading pre-trained models.
"""

import os
import pickle
import joblib
from typing import Optional, Any
import warnings


class ModelLoader:
    """
    Utility class for loading pre-trained models.
    
    Supports multiple model formats:
    - scikit-learn models (joblib/pickle)
    - PyTorch models (.pth, .pt)
    - TensorFlow/Keras models (.h5, .pb)
    """
    
    @staticmethod
    def load_sklearn_model(model_path: str) -> Any:
        """
        Load scikit-learn model from file.
        
        Args:
            model_path: Path to model file
            
        Returns:
            Loaded model object
        """
        try:
            # Try joblib first (preferred for sklearn)
            return joblib.load(model_path)
        except:
            # Fallback to pickle
            with open(model_path, 'rb') as f:
                return pickle.load(f)
    
    @staticmethod
    def load_pytorch_model(model_path: str, device: str = 'cpu'):
        """
        Load PyTorch model from file.
        
        Args:
            model_path: Path to model file
            device: Device to load model on ('cpu' or 'cuda')
            
        Returns:
            Loaded model object
        """
        try:
            import torch
            model = torch.load(model_path, map_location=device)
            if hasattr(model, 'eval'):
                model.eval()
            return model
        except ImportError:
            raise ImportError("PyTorch is required to load PyTorch models")
    
    @staticmethod
    def load_tensorflow_model(model_path: str):
        """
        Load TensorFlow/Keras model from file.
        
        Args:
            model_path: Path to model file
            
        Returns:
            Loaded model object
        """
        try:
            import tensorflow as tf
            return tf.keras.models.load_model(model_path)
        except ImportError:
            raise ImportError("TensorFlow is required to load TensorFlow models")
    
    @staticmethod
    def load_model(model_path: str, model_type: Optional[str] = None) -> Any:
        """
        Auto-detect and load model from file.
        
        Args:
            model_path: Path to model file
            model_type: Model type ('sklearn', 'pytorch', 'tensorflow') or None for auto-detect
            
        Returns:
            Loaded model object
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Auto-detect model type from extension
        if model_type is None:
            ext = os.path.splitext(model_path)[1].lower()
            if ext in ['.pkl', '.joblib']:
                model_type = 'sklearn'
            elif ext in ['.pth', '.pt']:
                model_type = 'pytorch'
            elif ext in ['.h5', '.pb']:
                model_type = 'tensorflow'
            else:
                # Default to sklearn
                model_type = 'sklearn'
        
        if model_type == 'sklearn':
            return ModelLoader.load_sklearn_model(model_path)
        elif model_type == 'pytorch':
            return ModelLoader.load_pytorch_model(model_path)
        elif model_type == 'tensorflow':
            return ModelLoader.load_tensorflow_model(model_path)
        else:
            raise ValueError(f"Unknown model type: {model_type}")


def load_pretrained_model(model_path: str, model_type: Optional[str] = None) -> Any:
    """
    Convenience function to load a pre-trained model.
    
    Args:
        model_path: Path to model file
        model_type: Model type or None for auto-detect
        
    Returns:
        Loaded model object
    """
    return ModelLoader.load_model(model_path, model_type)
