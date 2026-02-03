# AI Coding Instructions for Credit Risk Modeling

## Project Overview
Binary classification project to identify loan default risk using scikit-learn, LightGBM, XGBoost, and advanced preprocessing techniques. Uses Cookiecutter Data Science template with strict data pipeline separation (raw → interim → processed).

## Architecture & Data Pipeline

### Core Data Flow
1. **Raw Data** (`data/raw/`) → `credit_risk_modeling/dataset.py` → **Interim** (`data/interim/`)
2. **Interim Data** → `credit_risk_modeling/features.py` → **Processed** (`data/processed/`)
3. **Processed Data** → `credit_risk_modeling/modeling/{train.py, predict.py}`

### Key Modules
- **[config.py](credit_risk_modeling/config.py)**: Centralized path management using `pathlib.Path` and project root auto-discovery
- **[dataset.py](credit_risk_modeling/dataset.py)**: CLI tool (Typer-based) for data loading/preprocessing
- **[features.py](credit_risk_modeling/features.py)**: CLI tool for feature engineering from interim data
- **[model_eval.py](credit_risk_modeling/model_eval.py)**: Comprehensive model evaluation utilities (heavy sklearn/LightGBM usage)
- **[modeling/train.py](credit_risk_modeling/modeling/train.py)**: CLI tool for model training with hyperparameter tuning
- **[modeling/predict.py](credit_risk_modeling/modeling/predict.py)**: CLI tool for inference on test data

### CLI Execution Pattern
All data processing and modeling scripts use Typer for CLI. Run via:
```bash
python credit_risk_modeling/dataset.py
python credit_risk_modeling/features.py
python credit_risk_modeling/modeling/train.py
python credit_risk_modeling/modeling/predict.py
```

## Developer Workflows

### Environment Setup
```bash
make create_environment    # Create conda env from environment.yml
make requirements          # Update existing env with dependencies
```

### Code Quality
```bash
make format               # Fix all formatting with ruff (includes import sorting)
make lint                 # Check formatting and style with ruff
make test                 # Run pytest from tests/ directory
make clean                # Remove __pycache__ and .pyc files
make data                 # Execute data pipeline
```

## Project-Specific Conventions

### Import Standards
- **Ruff is configured** with import sorting (isort extension)
- First-party module: `known-first-party = ["credit_risk_modeling"]`
- Line length: 99 characters
- Always import `config` in modules to ensure path initialization

### Logging & Progress
- Use `loguru` logger instead of stdlib logging: `from loguru import logger`
- Use `tqdm` for progress bars in long-running operations
- Config automatically routes loguru output through tqdm when available

### Path Management
All data/model paths must use `credit_risk_modeling.config`:
```python
from credit_risk_modeling.config import (
    RAW_DATA_DIR,
    INTERIM_DATA_DIR, 
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    FIGURES_DIR
)
```
Never use hardcoded paths or `os.getcwd()`.

### Typer CLI Pattern
Data pipeline tools follow this structure:
- Single `app = typer.Typer()` instance
- One `@app.command()` main function with Path parameters
- Default paths from config module
- Clear logger messages: `.info()` for steps, `.success()` for completion

## Dependencies & Integration Points

### Key Libraries (from environment.yml)
- **Preprocessing**: `feature_engine`, `imbalanced-learn`, `pyampute`
- **Modeling**: `scikit-learn>=1.3`, `lightgbm`, `xgboost`, `statsmodels`
- **Class Imbalance**: `imbalanced-learn` (SMOTE available in model_eval)
- **Evaluation**: Rich sklearn metrics in `model_eval.py` (ROC, PR curves, calibration, etc.)
- **Visualization**: `matplotlib`, `seaborn`, `missingno`

### Data Files Structure
- **Raw**: `credit_risk_dataset.csv` (immutable source)
- **Interim**: Partially processed (e.g., `credit_risk_dataset_prepped.csv`, train/test splits as `y_train.csv`, `y_test.csv`)
- **Processed**: Model-ready features (`X_train_*.csv` and `X_test_*.csv` per model type: linear, tree, distance, neural, probability)

### Model Storage
- Trained models saved to `models/` directory
- Use `joblib` for model serialization (preferred in existing code)
- Predictions output to `data/processed/` as CSV files

## Code Patterns to Follow

### Data Processing (dataset.py, features.py)
```python
@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "dataset.csv",
    output_path: Path = INTERIM_DATA_DIR / "processed.csv",
):
    logger.info("Processing...")
    for item in tqdm(items, total=len(items)):
        # Process item
    logger.success("Complete.")
```

### Model Training (train.py)
- Use HalvingRandomSearchCV or HalvingGridSearchCV for hyperparameter tuning
- Apply pipelines with ColumnTransformer for mixed preprocessing
- Handle class imbalance (SMOTE available in model_eval)
- Use cross-validation with proper train/test separation

### Model Evaluation (model_eval.py)
- Comprehensive metrics: confusion matrix, ROC, PR, calibration curves
- Use `get_model_label()` for model identification in comparisons
- Support threshold tuning with `TunedThresholdClassifierCV`
- Multiple evaluation plots (confusion matrix, DET curve, partial dependence)

## Testing & Notebooks

### Testing
- Pytest configured in `environment.yml` (7.2.2)
- Tests located in `tests/` directory
- Run with `make test` or `pytest`
- Note: Existing test marked as failing (`assert False`)

### Notebooks
Located in `notebooks/` - use for prototyping and EDA:
- Distance, linear, neural, probability, tree model prototyping
- Preprocessing and EDA exploration
- Name new notebooks: `{number}-{description}.ipynb`

## Git & Documentation
- mkdocs configured for documentation (see `docs/mkdocs.yml`)
- License: MIT (see LICENSE)
- README provides full template structure overview
