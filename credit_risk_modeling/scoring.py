import json
import joblib
import numpy as np
from pathlib import Path
from loguru import logger
from credit_risk_modeling.config import MODELS_DIR

def load_model_and_preprocessor():
    """Load the best trained model and preprocessing pipeline from models/ directory."""
    try:
        model = joblib.load(MODELS_DIR / 'best_tree_model.pkl')
        preprocessor = joblib.load(MODELS_DIR / 'tree_preprocessed_pipeline.pkl')
        logger.info("✓ Model and preprocessor loaded successfully")
        return model, preprocessor
    except FileNotFoundError as e:
        logger.error(f"✗ Model files not found: {e}")
        raise

def calculate_pd_from_raw(features: dict, model, preprocessor) -> float:
    """
    Calculate Probability of Default (PD) from RAW applicant features (11 features).
    
    PRODUCTION USE: Pass in raw applicant data (11 features).
    The preprocessor will transform to 19 features, then model predicts.
    
    Args:
        features: dict with 11 raw features
                  {'person_age': 25, 'loan_amnt': 35000, ...}
        model: Trained LightGBM classifier
        preprocessor: Fitted preprocessing pipeline (11→19 features)
    
    Returns:
        float: Probability of default between 0-1
    """
    import pandas as pd
    
    # Convert dict to DataFrame (11 raw features)
    X = pd.DataFrame([features])
    
    # Apply preprocessing pipeline: 12 raw → 19 preprocessed features
    try:
        X_preprocessed = preprocessor.transform(X)
    except Exception as e:
        logger.error(f"Preprocessing failed for raw features: {e}")
        raise
    
    # Get probability of default from model (expects 19 features)
    try:
        pd_value = model.predict_proba(X_preprocessed, predict_disable_shape_check=True)[0, 1]
    except TypeError:
        # Fallback if predict_disable_shape_check not supported
        pd_value = model.predict_proba(X_preprocessed)[0, 1]
    
    logger.debug(f"Calculated PD: {pd_value:.4f} from raw features")
    return float(pd_value)


def calculate_pd_from_preprocessed(X_preprocessed, model) -> float:
    """
    Calculate Probability of Default (PD) from PREPROCESSED features (19 features).
    
    TESTING/VALIDATION USE: Pass in already-preprocessed features (19 features).
    Skips preprocessing step.
    
    Args:
        X_preprocessed: DataFrame or array with 19 preprocessed features
        model: Trained LightGBM classifier
    
    Returns:
        float: Probability of default between 0-1
    """
    import pandas as pd
    
    # Ensure it's a DataFrame with proper shape (1 row)
    if isinstance(X_preprocessed, dict):
        X_preprocessed = pd.DataFrame([X_preprocessed])
    elif not isinstance(X_preprocessed, pd.DataFrame):
        X_preprocessed = pd.DataFrame(X_preprocessed)
    
    # X_preprocessed should have exactly 19 columns
    if X_preprocessed.shape[1] != 19:
        logger.warning(f"Expected 19 preprocessed features, got {X_preprocessed.shape[1]}")
    
    # Get probability of default from model (already preprocessed)
    try:
        pd_value = model.predict_proba(X_preprocessed, predict_disable_shape_check=True)[0, 1]
    except TypeError:
        # Fallback if predict_disable_shape_check not supported
        pd_value = model.predict_proba(X_preprocessed)[0, 1]
    
    logger.debug(f"Calculated PD: {pd_value:.4f} from preprocessed features")
    return float(pd_value)

def calculate_ead(features: dict) -> float:
    """
    Calculate Exposure at Default (EAD) = loan amount.
    
    Args:
        features: dict with 'loan_amnt' key (or 'numeric__loan_amnt' if preprocessed)
    
    Returns:
        float: Loan amount in dollars
    """
    # Try both raw and preprocessed column names
    ead = features.get('loan_amnt') or features.get('numeric__loan_amnt', 0.0)
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
        return 1.0  # Assume 100% loss for all defaulted loans (Phase 2 baseline)
    
    elif phase == 3:
        try:
            lgd_path = MODELS_DIR / 'lgd_by_segment.json'
            with open(lgd_path, 'r') as f:
                lgd_map = json.load(f)
            return lgd_map.get(loan_intent, 1.0)
        except FileNotFoundError:
            logger.warning(f"LGD file not found, defaulting to 1.0")
            return 1.0
    
    else:
        logger.error(f"Unknown phase: {phase}")
        return 1.0

def calculate_expected_loss(pd: float, lgd: float, ead: float) -> float:
    """
    Calculate Expected Loss (EL) using the credit risk formula.
    
    EL = PD × LGD × EAD
    
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
    # Non-linear scaling: emphasize differences in high-PD range
    risk_score = int(np.clip(pd * 100, 0, 100))
    
    if risk_score < 33:
        risk_tier = 'LOW'
    elif risk_score < 67:
        risk_tier = 'MEDIUM'
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

def score_applicant(features: dict, phase: int = 2) -> dict:
    """
    Score a single applicant with full PD/LGD/EAD/EL components.
    
    PRODUCTION FUNCTION: Pass in raw applicant features (12 features).
    This is the main function called by both the API and batch processing.
    
    Args:
        features: dict of 12 raw applicant features
                  {'person_age': 25, 'loan_intent': 'PERSONAL', ...}
        phase: 2 (simplified LGD=1.0) or 3 (segment-specific LGD)
    
    Returns:
        dict: {
            'pd': float (0-1),
            'lgd': float (0-1),
            'ead': float ($),
            'expected_loss': float ($),
            'risk_score': int (0-100),
            'risk_tier': str ('LOW'|'MEDIUM'|'HIGH'),
            'confidence': float (0-1)
        }
    """
    try:
        model, preprocessor = load_model_and_preprocessor()
        
        # Calculate risk components - using raw features
        pd = calculate_pd_from_raw(features, model, preprocessor)
        ead = calculate_ead(features)
        loan_intent = features.get('loan_intent', 'UNKNOWN')
        lgd = get_lgd_by_segment(loan_intent, phase=phase)
        el = calculate_expected_loss(pd, lgd, ead)
        
        # Calculate business-ready outputs
        risk_score, risk_tier = normalize_to_risk_score(pd)
        confidence = calculate_confidence(pd)
        
        return {
            'pd': pd,
            'lgd': lgd,
            'ead': ead,
            'expected_loss': el,
            'risk_score': risk_score,
            'risk_tier': risk_tier,
            'confidence': confidence
        }
    
    except Exception as e:
        logger.error(f"Scoring failed: {e}")
        raise
