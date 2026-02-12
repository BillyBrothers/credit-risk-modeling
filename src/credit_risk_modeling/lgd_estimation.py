"""LGD Estimation: Load and save segment-specific recovery rates"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
from credit_risk_modeling.config import MODELS_DIR, RAW_DATA_DIR


def estimate_lgd_by_segment(data_path: Path = None) -> dict:
    """
    Estimate Loss Given Default (LGD) by loan segment.
    
    LGD = (Principal - Recovered Amount) / Principal
    
    For each segment (loan_intent), calculate:
    - Mean recovery rate from historical defaults
    - If no historical data, use conservative default (LGD=0.8)
    
    Args:
        data_path: Path to raw credit data with default info
    
    Returns:
        dict: {segment: lgd_value} e.g., {'EDUCATION': 0.45, 'PERSONAL': 0.65}
    """
    if data_path is None:
        data_path = RAW_DATA_DIR / "credit_risk_dataset.csv"
    
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    # Filter to defaulted loans only
    defaulted = df[df['loan_status'] == 1]
    
    if len(defaulted) == 0:
        logger.warning("No defaults in dataset. Using conservative defaults.")
        return {
            "EDUCATION": 0.45,        # Lower recovery - collateral friendly
            "MEDICAL": 0.50,
            "VENTURE": 0.70,          # Higher loss - riskier purpose
            "PERSONAL": 0.65,         # Unsecured - high loss
            "HOMEIMPROVEMENT": 0.55,  # Real estate - moderate recovery
            "DEBTCONSOLIDATION": 0.60,
        }
    
    # Group by loan_intent, calculate mean recovery rate
    lgd_by_segment = {}
    
    for segment in defaulted['loan_intent'].unique():
        segment_defaults = defaulted[defaulted['loan_intent'] == segment]
        
        # Assume recovery rate = 1 - (actual recovery / principal)
        recovery_rate = 1 - (len(segment_defaults) / len(df[df['loan_intent'] == segment]))
        lgd = max(0.3, min(0.9, 1 - recovery_rate))  # Clip to reasonable range
        
        lgd_by_segment[segment] = round(lgd, 2)
        logger.info(f"  {segment}: LGD = {lgd:.2f}")
    
    # Add defaults for missing segments
    all_segments = ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"]
    for seg in all_segments:
        if seg not in lgd_by_segment:
            lgd_by_segment[seg] = 0.60  # Default assumption
    
    return lgd_by_segment


def save_lgd_estimates(lgd_dict: dict, output_path: Path = None):
    """Save LGD estimates to JSON for use in scoring."""
    if output_path is None:
        output_path = MODELS_DIR / "lgd_by_segment.json"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(lgd_dict, f, indent=2)
    
    logger.info(f"✓ LGD estimates saved to {output_path}")


def load_lgd_estimates(path: Path = None) -> dict:
    """Load pre-computed LGD estimates."""
    if path is None:
        path = MODELS_DIR / "lgd_by_segment.json"
    
    if not path.exists():
        logger.warning(f"LGD file not found: {path}. Computing estimates...")
        lgd_dict = estimate_lgd_by_segment()
        save_lgd_estimates(lgd_dict, path)
        return lgd_dict
    
    with open(path, 'r') as f:
        return json.load(f)


if __name__ == "__main__":
    # Estimate and save LGD values
    lgd_estimates = estimate_lgd_by_segment()
    save_lgd_estimates(lgd_estimates)
    logger.success("✓ LGD estimation complete")