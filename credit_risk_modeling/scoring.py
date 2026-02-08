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
    Calculate Exposure at Default (EAD) = loan amount.

    Args:
        features: dict with "loan_amnt" key

    Returns:
        float: Loan amount in dollars
    """
    ead = features.get('loan_amnt', 0.0)
    if ead <= 0:
        logger.warning(f"Invalid EAD: {ead}")
        return 0.0
    return float(ead)

def get_lgd_by_segment(loan_intent: str, phase: int = 2) -> float:
    """
    Get Loss Given Default (LGD) by loan segment.

    Phase 2: Return 1.0 (100% loss - simplified baseline)
    Phase 3: Load from lgd_by_segment.json (recovered-amount-based)

    Args:
        loan_intent: Type of loan (PERSONAL, EDUCATION, MEDICAL, etc.)
        phase: 2 for simplified, 3 for segment-specific

    Returns:
        float: LGD between 0-1
    """

    if phase == 2:
        return 1.0 # Assume 100% loss for all defaulted loans (Phase 2 baseline)
    
    elif phase == 3:
        try:
            lgd_path = "../models/lgd_by_segment.json"
            with open(lgd_path, 'r') as f:
                lgd_map = json.load(f)
            return lgd_map.get(loan_intent, 1.0)
        except FileNotFoundError as e:
            logger.warning(f"LGD file not found, default to 1.0")
            return 1.0
    else:
        logger.error(f"Unknown phase: {phase}")
        return 1.0
    
def calculate_expected_loss(pd: float, lgd: float, ead: float) -> float:
    """
    Calculate Expected Loss (EL) using the credit risk formula.

    EL = PD x LGD x EAD

    Args:
        pd: Probability of Default (0-1)
        lgd: Loss Given Default (0-1)
        ead: Exposure at Default ($)

    Returns:
        float: Expected Loss in dollars
    """
    el = pd * lgd * ead
    return float(el)

def normalize_to_risk_score(pd: float) -> tuple:
    """
    Convert probability (0-1) to business-friendly risk score (0-100).
    Assign risk tier: LOW, MEDIUM, HIGH.

    Args:
        pd: Probability of Default (0-1)

    Returns:
        tuple: (risk_score: int 0-100, risk_tier: str)
    """

    risk_score = int(np.clip(a = pd * 100, a_min = 0, a_max = 100))

    if risk_score < 33:
        risk_tier = 'LOW'
    elif risk_score < 67:
        risk_tier = "MEDIUM"
    else:
        risk_tier = 'HIGH'
    
    return risk_score, risk_tier

def calculate_confidence(pd: float) -> float:
    """
    Calculate confidence score (how sure is the model?).
    Based on distance from 50% probability (most confident at 0% or 100%).

    Args:
        pd: Probability of Default (0-1)
    
    Returns:
        float: Confidence 0-1
    """
    
    confidence = 1 - abs(pd - 0.5) * 2
    return float(np.clip(confidence, 0, 1))

