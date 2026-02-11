"""FastAPI server for real-time applicant scoring and approval decisions"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from loguru import logger
import traceback

from credit_risk_modeling import scoring, decision_rules


