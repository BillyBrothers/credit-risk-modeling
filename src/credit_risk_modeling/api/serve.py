"""FastAPI server for real-time applicant scoring and approval decisions"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from loguru import logger
import traceback

from credit_risk_modeling import scoring, decision_rules

app = FastAPI(
    title = "Credit Risk Scoring API",
    description = "Real-time loan applicant risk scoring and approval decisions",
    version= "1.0.0"
)

# Load model and preprocessor on startup
model, preprocessor = scoring.load_model_and_processor()
engine = decision_rules.ApprovalRuleEngine()

# logger.info("✓ FastAPI server initialized with model and decision engine")

class ApplicantRequest(BaseModel):
    """Applicant features for scoring request"""
    person_age: int = Field(..., ge=18, le=90)
    person_income: int = Field(..., ge=0)
    person_home_ownership: str = Field(..., regex="^(RENT|OWN|MORTGAGE)$")
    person_emp_length: float = Field(..., ge=0)
    loan_intent: str = Field(..., regex="^(EDUCATION|MEDICAL|VENTURE|PERSONAL|HOMEIMPROVEMENT|DEBTCONSOLIDATION)$")
    loan_grade: str = Field(..., regex="^[A-G]$")
    loan_amnt: int = Field(..., ge=100, le=100000)
    loan_int_rate: float = Field(..., ge=0, le=35)
    loan_percent_income: float = Field(..., ge=0, le=1)
    cb_person_default_on_file: bool
    cb_person_cred_hist_length: int = Field(..., ge=0)


class ScoringResponse(BaseModel):
    """Scoring result with PD/LGD/EAD/EL"""
    pd: float
    lgd: float
    ead: float
    expected_loss: float
    risk_score: int
    risk_tier: str
    confidence: float

class ApprovalResponse(BaseModel):
    """Approval decision and reasoning"""
    decision: str
    reason: str
    el_pct: float

class FullScoringResponse(BaseModel):
    """Complete response: scoring + approval"""
    scoring: ScoringResponse
    approval: ApprovalResponse
    applicant_id: Optional[str] = None

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Credit Risk Scoring API"}


@app.post("/score", response_model=FullScoringResponse)
async def score_applicant(applicant: ApplicantRequest, phase: int = 2, applicant_id: Optional[str] = None):
    """
    Score a single applicant and return approval decision.
    
    Args:
        applicant: Applicant features (11 inputs)
        phase: 2 (LGD=1.0) or 3 (segment-specific LGD)
        applicant_id: Optional ID for tracking/audit
    
    Returns:
        Scoring metrics + approval decision
    """
    try:
        # Convert request to dict
        features = applicant.dict()
        
        # Score applicant
        score = scoring.score_applicant(features=features, phase=phase)
        
        # Get approval decision
        decision = engine.decide(
            pd=score['pd'],
            lgd=score['lgd'],
            ead=score['ead'],
            expected_loss=score['expected_loss'],
            loan_intent=features['loan_intent']
        )
        
        # Format response
        response = FullScoringResponse(
            scoring=ScoringResponse(**{k: v for k, v in score.items() if k in ['pd', 'lgd', 'ead', 'expected_loss', 'risk_score', 'risk_tier', 'confidence']}),
            approval=ApprovalResponse(**decision),
            applicant_id=applicant_id
        )
        
        logger.info(f"✓ Scored applicant {applicant_id or 'unknown'}: {decision['decision']}")
        return response
    
    except Exception as e:
        logger.error(f"✗ Scoring failed for applicant {applicant_id or 'unknown'}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Scoring failed: {str(e)}")

