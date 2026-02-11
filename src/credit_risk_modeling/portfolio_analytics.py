"""Portfolio analytics and risk aggregation"""
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
from dataclasses import dataclass

@dataclass
class PortfolioMetrics:
    """Aggregated portfolio statistics"""
    total_applicants: int
    approval_rate: float
    denial_rate: float
    manual_review_rate: float
    avg_pd: float
    avg_lgd: float
    avg_ead: float
    total_ead: float
    total_expected_loss: float
    
    # Segmentation
    by_risk_tier: dict
    by_loan_intent: dict
    by_decision: dict