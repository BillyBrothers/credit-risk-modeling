import json
import joblib
import numpy as np
from pathlib import Path
from loguru import logger
from credit_risk_modeling.config import MODELS_DIR


def load_model_and_preprocessor():
    """Load the best trained model and preprocessing steps from models/ directory"""
    try:
        model = joblib.load(filename= "../models/best_tree_model.pkl")
        preprocessor = joblib.load(filename="../models/tree_processed_pipeline.pkl")
        logger.info("Model and preprocessor loaded successfully")
        return model, preprocessor
    except FileNotFoundError as e:
        logger.error(f"Model files not found: {e}")
        raise

def calculate_pd(features: dict, model, preprocessor) -> float:
    """Calculate Probability of Default (PD) using the trained LightGBM Model.

    Args:
        features: dict of applicant features (raw or preprocesed)
        model: Trained LightGBM Classifier
        preprocessor: Fitted preprocessing pipeline

    Returns:
        float: Probability between 0-1

    """
    import pandas as pd

    # Expected by preprocessor
    X =  pd.DataFrame([features])

    try:
        X_preprocessed = preprocessor.transform(X)
    except Exception as e:
        logger.warning(f"Preprocessing failed, attempting direct prediction: {e}")
        X_preprocessed = X
    
    pd_value = model.predict_proba(X_preprocessed)[0, 1]
    return float(pd_value)

def calculate_ead(features: dict) -> float:
    """
    Calculate Exposure at Default (EAD) = loan amount
    """
    