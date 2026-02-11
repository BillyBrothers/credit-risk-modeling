from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from credit_risk_modeling.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()

@app.command()
def batch_predict(
    input_path: Path = typer.Option(
        config.PROCESSED_DATA_DIR / "applicants.csv",
        help="Path to CSV file with applicant features"
    ),
    output_path: Path = typer.Option(
        config.PROCESSED_DATA_DIR / "scoring_results.csv",
        help="Path to output CSV with scores and decisions"
    ),
    phase: int = typer.Option(2, help="LGD phase: 2 (simplified) or 3 (segment-specific)"),
    include_reasoning: bool = typer.Option(True, help="Include approval reasoning in output")
):

    """
    Score multiple applicants from CSV file.
    
    Input CSV must have 11 columns (raw features):
    person_age, person_income, person_home_ownership, person_emp_length,
    loan_intent, loan_grade, loan_amnt, loan_int_rate, loan_percent_income,
    cb_person_default_on_file, cb_person_cred_hist_length
    
    Output CSV includes: [input features] + [pd, lgd, ead, expected_loss, 
    risk_score, risk_tier, confidence, decision, el_pct] + [reason]
    """
try:
        # Load input data
        logger.info(f"Loading applicants from {input_path}")
        df = pd.read_csv(input_path)
        logger.info(f"✓ Loaded {len(df)} applicants")
        
        # Load model
        model, preprocessor = scoring.load_model_and_preprocessor()
        engine = decision_rules.ApprovalRuleEngine()
        
        # Score each applicant
        results = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Scoring"):
            try:
                features = row.to_dict()
                
                # Score
                score = scoring.score_applicant(features=features, phase=phase)
                
                # Decide
                decision = engine.decide(
                    pd=score['pd'],
                    lgd=score['lgd'],
                    ead=score['ead'],
                    expected_loss=score['expected_loss'],
                    loan_intent=features.get('loan_intent', 'UNKNOWN')
                )
                
                # Combine results
                result = {
                    **features,
                    'pd': score['pd'],
                    'lgd': score['lgd'],
                    'ead': score['ead'],
                    'expected_loss': score['expected_loss'],
                    'risk_score': score['risk_score'],
                    'risk_tier': score['risk_tier'],
                    'confidence': score['confidence'],
                    'decision': decision['decision'],
                    'el_pct': decision['el_pct']
                }
                
                if include_reasoning:
                    result['approval_reason'] = decision['reason']
                
                results.append(result)
            
            except Exception as e:
                logger.warning(f"Failed to score applicant {idx}: {str(e)}")
                continue



@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = PROCESSED_DATA_DIR / "test_features.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Performing inference for model...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Inference complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
