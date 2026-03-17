# Pre-trained Models Directory

This directory contains pre-trained machine learning models for ADMET prediction.

## Model Files

Each predictor can use a pre-trained model file. Model files should be named as:
- `metabolism_model.pkl` - Metabolism prediction model
- `plasma_exposure_model.pkl` - Plasma exposure prediction model
- `bbb_model.pkl` - BBB permeability prediction model
- `organ_toxicity_model.pkl` - Organ toxicity prediction model
- `tcm_model.pkl` - TCM prediction model
- `half_life_model.pkl` - Half-life prediction model

## Model Formats

Models can be in the following formats:
- **scikit-learn models**: `.pkl` or `.joblib` files
- **PyTorch models**: `.pth` or `.pt` files
- **TensorFlow/Keras models**: `.h5` or `.pb` files

## Model Loading

Models are automatically loaded by the predictors if:
1. `model_path` is provided during predictor initialization
2. `model_dir` is provided to `EnsembleADMETPredictor` and model files exist

## Default Behavior

If no model files are found, predictors will use rule-based approaches as fallback.

## Training Custom Models

To train custom models:
1. Prepare training data with molecular descriptors and labels
2. Train models using scikit-learn, PyTorch, or TensorFlow
3. Save models in the appropriate format
4. Place model files in this directory or specify custom paths

## Model Requirements

- Models should accept feature vectors from `DescriptorExtractor`
- Classification models should implement `predict()` and optionally `predict_proba()`
- Regression models should implement `predict()` and return continuous values
