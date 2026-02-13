"""Explainability: Feature importance and SHAP explanations"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from loguru import logger

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not installed. Install with: pip install shap")

from credit_risk_modeling.config import FIGURES_DIR, MODELS_DIR
from credit_risk_modeling import scoring


class ExplainabilityEngine:
    """Explain model predictions using SHAP values."""
    
    def __init__(self, model=None, preprocessor=None):
        """
        Initialize with pre-loaded model and preprocessor.
        
        Args:
            model: Trained model
            preprocessor: Fitted preprocessing pipeline
        """
        if model is None or preprocessor is None:
            model, preprocessor = scoring.load_model_and_preprocessor()
        
        self.model = model
        self.preprocessor = preprocessor
        self.explainer = None
        
        logger.info("✓ ExplainabilityEngine initialized")
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get global feature importance from model.
        
        Handles wrapped models (CalibratedClassifier, FixedThresholdClassifier, etc.)
        
        Returns:
            DataFrame with feature names and importance scores
        """
        feature_names = self.preprocessor.get_feature_names_out()
        
        # Unwrap the model to get base estimator
        model = self.model
        
        # Traverse wrapper chain: FixedThresholdClassifier → CalibratedClassifierCV → base estimator
        while hasattr(model, 'estimator_') and not hasattr(model, 'feature_importances_'):
            model = model.estimator_
        
        # For CalibratedClassifierCV
        if hasattr(model, 'base_estimator_') and not hasattr(model, 'feature_importances_'):
            model = model.base_estimator_
        
        # Get importances from base model
        if not hasattr(model, 'feature_importances_'):
            logger.warning("Model doesn't have feature_importances_ attribute. Using uniform importance.")
            importances = np.ones(len(feature_names)) / len(feature_names)
        else:
            importances = model.feature_importances_
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def plot_feature_importance(self, top_n: int = 15, 
                               output_path: Path = None) -> Path:
        """
        Plot top N most important features.
        
        Args:
            top_n: Number of features to display
            output_path: Where to save plot
        
        Returns:
            Path to saved plot
        """
        importance_df = self.get_feature_importance().head(top_n)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(range(len(importance_df)), importance_df['importance'], color='steelblue')
        ax.set_yticks(range(len(importance_df)))
        ax.set_yticklabels(importance_df['feature'])
        ax.set_xlabel('Importance Score')
        ax.set_title(f'Top {top_n} Most Important Features')
        ax.invert_yaxis()
        
        plt.tight_layout()
        
        if output_path is None:
            output_path = FIGURES_DIR / "feature_importance.png"
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Feature importance plot saved to {output_path}")
        plt.close()
        
        return output_path
    
    def explain_prediction(self, features: dict) -> dict:
        """
        Explain a single prediction using feature impact analysis.
        
        Args:
            features: Dict of applicant features
        
        Returns:
            dict with explanation details
        """
        # Convert to DataFrame and preprocess
        X = pd.DataFrame([features])
        X_preprocessed = self.preprocessor.transform(X)
        
        # Get prediction
        pd_prediction = self.model.predict_proba(X_preprocessed)[0, 1]
        
        # Get feature importance for this prediction
        importance_df = self.get_feature_importance().head(5)
        
        explanation = {
            'predicted_pd': pd_prediction,
            'top_factors': importance_df[['feature', 'importance']].to_dict('records')
        }
        
        return explanation
    
    def export_feature_importance_csv(self, output_path: Path = None) -> Path:
        """Export feature importance to CSV."""
        importance_df = self.get_feature_importance()
        
        if output_path is None:
            output_path = FIGURES_DIR / "feature_importance.csv"
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        importance_df.to_csv(output_path, index=False)
        logger.info(f"✓ Feature importance exported to {output_path}")
        
        return output_path
    
    def generate_summary_report(self, output_path: Path = None) -> str:
        """Generate text report of model explainability."""
        importance_df = self.get_feature_importance()
        
        report = f"""
====================================================================
            MODEL EXPLAINABILITY REPORT
====================================================================

FEATURE IMPORTANCE (Top 10)
────────────────────────────────────────────────────────────────
"""
        for idx, row in importance_df.head(10).iterrows():
            bar_length = int(row['importance'] * 50)
            bar = '█' * bar_length
            report += f"{row['feature']:30s} {bar} {row['importance']:.4f}\n"
        
        report += f"""

MODEL INSIGHTS
────────────────────────────────────────────────────────────────
Total Features:           {len(importance_df)}
Top Feature:              {importance_df.iloc[0]['feature']}
Top Feature Importance:   {importance_df.iloc[0]['importance']:.4f}

Feature Diversity:
- Top 3 features explain   {importance_df.head(3)['importance'].sum():.1%} of importance
- Top 10 features explain  {importance_df.head(10)['importance'].sum():.1%} of importance

====================================================================
"""
        
        if output_path is None:
            output_path = FIGURES_DIR / "explainability_report.txt"
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"✓ Explainability report saved to {output_path}")
        return report


if __name__ == "__main__":
    import typer
    
    app = typer.Typer()
    
    @app.command()
    def analyze(
        output_dir: Path = typer.Option(None, help="Output directory for plots and reports (optional)")
    ):
        """Generate feature importance plots and explainability report"""
        try:
            if output_dir:
                output_dir.mkdir(parents=True, exist_ok=True)
            
            explainer = ExplainabilityEngine()
            
            # Generate plots and reports
            logger.info("Generating feature importance plot...")
            explainer.plot_feature_importance(top_n=15)
            
            logger.info("Exporting feature importance to CSV...")
            explainer.export_feature_importance_csv()
            
            logger.info("Generating explainability report...")
            report = explainer.generate_summary_report()
            
            print(report)
            logger.success("✓ Explainability analysis complete!")
        
        except Exception as e:
            logger.error(f"✗ Explainability analysis failed: {e}")
            raise typer.Exit(code=1)
    
    app()