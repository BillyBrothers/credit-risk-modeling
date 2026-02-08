from dataclasses import dataclass, field
from loguru import logger

@dataclass
class ApprovalRules:
    """Configuration for approval thresholds (easy to change for A/B testing)."""
    auto_approve_el_pct: float = 0.05      # EL < 5% of EAD → APPROVE
    manual_review_el_pct: float = 0.15     # 5% < EL < 15% → MANUAL_REVIEW
                                            # EL >= 15% → DENY
    
    segment_overrides: dict = field(default_factory=lambda: {
        'EDUCATION': {'auto_approve': 0.20, 'manual_review': 0.35},  # More lenient (gov-backed)
        'PERSONAL': {'auto_approve': 0.08, 'manual_review': 0.12},   # Stricter (higher risk)
        'MEDICAL': {'auto_approve': 0.10, 'manual_review': 0.20},    # Moderate
    })

class ApprovalRuleEngine:
    """Orchestrates loan approval decisions based on credit risk metrics."""
    
    def __init__(self, rules: ApprovalRules = None):
        """
        Initialize the approval engine.
        
        Args:
            rules: ApprovalRules config. If None, uses defaults.
        """
        self.rules = rules or ApprovalRules()
        logger.info("✓ ApprovalRuleEngine initialized")
    
    def decide(self, pd: float, lgd: float, ead: float, expected_loss: float, 
               loan_intent: str = None) -> dict:
        """
        Make approval decision based on credit risk components.
        
        Args:
            pd: Probability of Default (0-1)
            lgd: Loss Given Default (0-1)
            ead: Exposure at Default ($)
            expected_loss: Expected Loss in dollars
            loan_intent: Loan type (for segment-specific thresholds)
        
        Returns:
            dict: {
                'decision': 'APPROVE'|'MANUAL_REVIEW'|'DENY',
                'reason': str (explanation for audit trail),
                'el_pct': float (EL as % of EAD)
            }
        """
        # Calculate EL as percentage of EAD
        el_pct = expected_loss / ead if ead > 0 else 1.0
        
        # Get thresholds (use segment override if available)
        if loan_intent and loan_intent in self.rules.segment_overrides:
            auto_approve_threshold = self.rules.segment_overrides[loan_intent]['auto_approve']
            manual_review_threshold = self.rules.segment_overrides[loan_intent]['manual_review']
            reason_prefix = f"[{loan_intent}] "
        else:
            auto_approve_threshold = self.rules.auto_approve_el_pct
            manual_review_threshold = self.rules.manual_review_el_pct
            reason_prefix = ""
        
        # Decision logic
        if el_pct < auto_approve_threshold:
            decision = 'APPROVE'
            reason = (
                f"{reason_prefix}Low expected loss: "
                f"${expected_loss:,.0f} ({el_pct*100:.1f}% of ${ead:,.0f}). "
                f"PD={pd*100:.1f}%, LGD={lgd*100:.0f}%"
            )
        
        elif el_pct < manual_review_threshold:
            decision = 'MANUAL_REVIEW'
            reason = (
                f"{reason_prefix}Moderate expected loss: "
                f"${expected_loss:,.0f} ({el_pct*100:.1f}% of ${ead:,.0f}). "
                f"Manual review required. PD={pd*100:.1f}%, LGD={lgd*100:.0f}%"
            )
        
        else:
            decision = 'DENY'
            reason = (
                f"{reason_prefix}High expected loss: "
                f"${expected_loss:,.0f} ({el_pct*100:.1f}% of ${ead:,.0f}). "
                f"Risk unacceptable. PD={pd*100:.1f}%, LGD={lgd*100:.0f}%"
            )
        
        return {
            'decision': decision,
            'reason': reason,
            'el_pct': el_pct
        }
