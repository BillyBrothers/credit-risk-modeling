"""FastAPI server for real-time applicant scoring and approval decisions"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from loguru import logger
import traceback

from credit_risk_modeling import scoring, decision_rules

FastAPI(
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
