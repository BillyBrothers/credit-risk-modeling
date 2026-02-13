# Credit Risk Modeling

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

**Production-ready credit risk modeling system** for binary classification of loan default risk. Combines supervised learning (LightGBM, XGBoost), advanced preprocessing, real-time APIs, and comprehensive risk analytics.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         credit_risk_modeling and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── credit_risk_modeling   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes credit_risk_modeling a Python module
    │
    ├── config.py               <- Centralized path management and configuration
    │
    ├── dataset.py              <- CLI: Load raw data and create interim datasets
    │
    ├── preprocessing.py        <- CLI: Feature engineering and model-specific pipelines
    │
    ├── features.py             <- CLI: Feature engineering from interim data
    │
    ├── model_eval.py           <- Model evaluation utilities (ROC, PR curves, calibration)
    │
    ├── scoring.py              <- Risk scoring engine (PD, LGD, EAD, Expected Loss)
    │
    ├── decision_rules.py       <- Approval decision orchestration
    │
    ├── portfolio_analytics.py  <- Aggregated portfolio risk metrics and analysis
    │
    ├── risk_reporter.py        <- Executive risk dashboards and KPI reporting
    │
    ├── lgd_estimator.py        <- Segment-specific loss given default estimation
    │
    ├── model_monitor.py        <- Model drift detection and data quality monitoring
    │
    ├── explainibility.py       <- Feature importance and prediction explanability
    │
    ├── plots.py                <- Visualization utilities
    │
    ├── api/
    │   └── serve.py            <- FastAPI server for real-time scoring endpoints
    │
    └── modeling                
        ├── __init__.py 
        ├── predict.py          <- CLI: Batch scoring on CSV files          
        └── train.py            <- CLI: Model training with hyperparameter tuning
```

--------

## Quick Start

### Environment Setup

```bash
# Create conda environment
make create_environment

# Activate environment
conda activate credit_risk_modeling

# Install dependencies
make requirements
```

### Data Pipeline

The project follows a strict three-stage data pipeline:

**Stage 1: Raw → Interim** (Load & Basic Preprocessing)
```bash
python -m credit_risk_modeling.dataset
```
- Loads `data/raw/credit_risk_dataset.csv`
- Handles missing values, explores MCAR/MAR/MNAR patterns
- Creates train/val/test split
- Outputs: `data/interim/credit_risk_dataset_prepped.csv`, `y_train.csv`, `y_test.csv`, `y_val.csv`

**Stage 2: Interim → Processed** (Feature Engineering & Model-Specific Pipelines)
```bash
python -m credit_risk_modeling.features
python -m credit_risk_modeling.preprocessing
```
- Features: Feature engineering (binning, scaling, transformations)
- Preprocessing: Creates 5 model-specific feature sets (linear, tree, distance, probability, neural)
- Outputs: `data/processed/X_{train,test,val}_{linear,tree,distance,probability,neural}.csv`

**Stage 3: Processed → Models** (Training & Scoring)
```bash
python -m credit_risk_modeling.modeling.train
python -m credit_risk_modeling.modeling.predict --input-csv data/processed/test_data.csv
```

### Model Training

Train LightGBM, XGBoost, and scikit-learn models with hyperparameter tuning:

```bash
python -m credit_risk_modeling.modeling.train \
    --model-types linear tree probability \
    --cv-folds 5 \
    --n-iter 20
```

Outputs:
- Trained model pipelines to `models/`
- Cross-validation metrics and plots to `reports/figures/`

### Batch Scoring

Score new applicants from CSV:

```bash
python -m credit_risk_modeling.modeling.predict \
    --input-csv new_applicants.csv \
    --model-type tree
```

Outputs predictions with risk scores, tiers, and approval decisions to `data/processed/predictions.csv`

### Real-Time API Server

Launch FastAPI server for live scoring:

```bash
python -m credit_risk_modeling.api.serve
```

Endpoints:
- `GET /health` - Server status
- `POST /score` - Score single applicant
- `POST /batch-score` - Score multiple applicants

### Risk Scoring & Reporting

Generate executive risk dashboards:

```bash
python -m credit_risk_modeling.risk_reporter \
    --input-csv data/processed/test_results.csv
```

Outputs:
- HTML dashboard with KPIs to `reports/figures/risk_report_[timestamp].html`
- CSV metrics export

### Advanced Analytics

**Model Explainability**
```bash
python -m credit_risk_modeling.explainibility
```
Generates feature importance plots, rankings, and model insights.

**Model Monitoring**
```bash
python -m credit_risk_modeling.model_monitor
```
Detects performance drift, data quality issues, input distribution changes.

**Loss Given Default (LGD) Estimation**
```bash
python -m credit_risk_modeling.lgd_estimator
```
Computes segment-specific recovery rates for Basel risk calculations.

## Architecture

### Data Flow

```
Raw Data (CSV)
     ↓
[dataset.py] → Data Cleaning, Missing Value Analysis, Train/Test Split
     ↓
Interim Data (Cleaned, Split)
     ↓
[features.py] → Feature Engineering, Transformations
[preprocessing.py] → Model-Specific Pipelines
     ↓
Processed Data (5 Feature Sets for Different Model Types)
     ↓
[train.py] → Train LightGBM, XGBoost, scikit-learn Models
[predict.py] → Batch Inference
[serve.py] → Real-Time API
     ↓
[scoring.py] → Risk Score Calculation (PD × LGD × EAD = Expected Loss)
[decision_rules.py] → Approval Decisions
     ↓
[risk_reporter.py] → Executive Dashboards
[portfolio_analytics.py] → Portfolio Metrics
[model_monitor.py] → Performance Monitoring
[explainibility.py] → Feature Importance Analysis
```

### Risk Framework

**Basel III Credit Risk Formula:**
- **PD** (Probability of Default): ML model prediction
- **LGD** (Loss Given Default): Segment-specific recovery % from historical data
- **EAD** (Exposure at Default): Loan amount at time of default
- **Expected Loss** = PD × LGD × EAD

**Decision Logic:**
- **Auto-Approve**: EL < 5% AND EAD < threshold
- **Manual Review**: 5% ≤ EL ≤ 15% OR segment-specific rules apply
- **Deny**: EL > 15% OR high risk tier + low credit score

## Code Quality

```bash
make format               # Fix formatting with ruff
make lint                 # Check code style
make test                 # Run pytest suite
make clean               # Remove cache files
```

## Key Features

- **Production ML Stack**: LightGBM, XGBoost, scikit-learn with calibrated classifiers
- **Advanced Preprocessing**: Iterative imputation, power transformation, outlier handling
- **Model-Specific Pipelines**: Linear, tree, distance, probability, and neural network feature sets
- **Real-Time API**: FastAPI server with Pydantic validation
- **Risk Framework**: Basel III-aligned PD/LGD/EAD calculations
- **Executive Reporting**: HTML dashboards, CSV exports, KPI tracking
- **Model Monitoring**: Drift detection, data quality checks, performance tracking
- **Explainability**: Feature importance analysis, prediction explanations

## Project Structure

The project follows **Cookiecutter Data Science** conventions with strict data pipeline separation:
- **Raw** (`data/raw/`): Immutable source data
- **Interim** (`data/interim/`): Partially processed data with train/test splits
- **Processed** (`data/processed/`): Model-ready feature sets
- **Models** (`models/`): Trained pipelines and serialized estimators
- **Reports** (`reports/`): Analysis outputs, dashboards, visualizations

## Requirements

- Python 3.11+
- LightGBM, XGBoost, scikit-learn
- FastAPI, Uvicorn, Typer
- pandas, numpy, feature_engine
- Conda (for environment management)

See `environment.yml` for complete dependencies.

## License

MIT License - see LICENSE file for details.

--------

