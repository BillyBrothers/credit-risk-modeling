"""Decision Rules Engine: Business approval orchestration based on risk metrics"""
from dataclasses import dataclass
from loguru import logger
from typing import Optional


@dataclass
class ApprovalRules:
    """Configurable thresholds for approval decisions.
    
    Allows easy A/B testing of business policies without code changes.
    Thresholds are based on Expected Loss as % of EAD.
    """
    auto_approve_el_pct: float = 0.05  # < 5% EL/EAD = auto-approve
    manual_review_el_pct: float = 0.15  # 5-15% EL/EAD = manual review
    # > 15% EL/EAD = auto-deny (implicit)
    
    # Segment-specific overrides (optional soft rules, not hard cutoffs)
    segment_strategies: dict = None
    
    def __post_init__(self):
        if self.segment_strategies is None:
            self.segment_strategies = {
                "EDUCATION": {"auto_approve_el_pct": 0.06, "manual_review_el_pct": 0.18},  # More lenient
                "MEDICAL": {"auto_approve_el_pct": 0.05, "manual_review_el_pct": 0.15},
                "PERSONAL": {"auto_approve_el_pct": 0.04, "manual_review_el_pct": 0.13},  # Stricter
                "VENTURE": {"auto_approve_el_pct": 0.03, "manual_review_el_pct": 0.12},  # Very strict
            }

class ApprovalRuleEngine:
    """Orchestrates approval decisions based on risk metrics and business rules.
    
    Separates ML predictions (scoring.py) from business policy (this module).
    Enables audit trails and consistent decision-making across applicants.
    """

    def __init__(self, rules: ApprovalRules = None):
         """Initialize engine with optional custom rules.
        
        Args:
            rules: ApprovalRules dataclass with configurable thresholds.
                   If None, uses defaults.
        """       

        self.rules = rules or ApprovalRules()
        logger.info(f"✓ ApprovalRuleEngine initialized with thresholds: "
                    f"auto_approve={self.rules.auto_approve_el_pct*100:.0f} "
                    f"manual_review={self.rules.auto_approve_el_pct*100:.0f} ") 

def decide(self, pd: float, lgd: float, ead: float, expected_loss: float,
            loan_intent: str = "UNKNOWN") -> dict:
        """Make approval decision based on risk metrics and business rules.
            
            Args:
                pd: Probability of Default (0-1)
                lgd: Loss
                ead: Exposure at Default ($)
                expected_loss: Expected Loss in dollars
                loan_intent: Loan purpose (EDUCATION, MEDICAL, PERSONAL, VENTURE, etc.)

            Returns:
                dict with keys:
                    - 'decision': 'APPROVE', 'DENY', or 'MANUAL_REVIEW'
                    - 'reason': Explanation for audit trial
                    - 'el_pct': Expected Loss as % of EAD
            """

            # Calculate EL as % of EAD (key business metric)
            el_pct = expected_loss / ead if ead > 0 else 1.0

            # Get segment-specific thresholds if available
            segment_rules = self.rules.segment_strategies.get(loan_intent, {})
            auto_approve_threshold = segment_rules.get("auto_approve_el_pct", self.rules.auto_approve_el_pct)
            manual_review_threshold = segment_rules.get("manual_review_el_pct", self.rules.manual_review_el_pct)
            

            # Determine Decision
            if el_pct < auto_approve_threshold:
                decision = "APPROVE"
                reason = (f"Manual review required: Expected Loss of {el_pct*100:.2f}% is below"
                          f"auto-approve threshold ({auto_approve_threshold*100:.2f}%) for {loan_intent} loans."
                          f"PD={pd:.2%}, LGD={lgd:.0%}, EAD=${ead:,.0f}")

