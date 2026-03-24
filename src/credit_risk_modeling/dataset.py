"""
Dataset module for loading and preparing raw credit risk data.

Handles loading raw data, basic preprocessing (duplicates, missing values),
and creating train/validation/test splits for downstream modeling.
"""

from pathlib import Path

import pandas as pd
import typer
from loguru import logger
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from credit_risk_modeling.config import INTERIM_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


def load_raw_data(input_path: Path) -> pd.DataFrame:
    """Load raw dataset from CSV."""
    logger.info(f"Loading raw data from {input_path}")
    df = pd.read_csv(input_path)
    logger.info(f"Loaded dataset with shape {df.shape}")
    return df


def handle_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows from dataset."""
    initial_shape = df.shape
    df = df.drop_duplicates(keep="first")
    removed = initial_shape[0] - df.shape[0]
    if removed > 0:
        logger.info(f"Removed {removed} duplicate rows")
    return df


def create_splits(
    df: pd.DataFrame, test_size: float = 0.3, val_size: float = 0.5, random_state: int = 42
) -> tuple:
    """Create train/validation/test splits with stratification on target."""
    logger.info("Creating train/validation/test splits (70/15/15)...")

    # Extract features and target
    X = df.drop("loan_status", axis=1)
    y = df["loan_status"]

    # First split: train (70%) and temp (30%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Second split: val (15%) and test (15%) from temp
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=random_state, stratify=y_temp
    )

    logger.info(
        f"Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}"
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def save_splits(
    df: pd.DataFrame,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    output_dir: Path,
) -> None:
    """Save preprocessed dataset and splits to interim directory."""
    logger.info(f"Saving datasets to {output_dir}")

    # Create output directory if needed
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save preprocessed dataset
    prepped_path = output_dir / "credit_risk_dataset_prepped.csv"
    df.to_csv(prepped_path, index=False)
    logger.info(f"Saved preprocessed dataset to {prepped_path}")

    # Save feature sets
    X_train_path = output_dir / "X_train.csv"
    X_val_path = output_dir / "X_val.csv"
    X_test_path = output_dir / "X_test.csv"

    X_train.to_csv(X_train_path, index=False)
    X_val.to_csv(X_val_path, index=False)
    X_test.to_csv(X_test_path, index=False)

    logger.info(f"Saved feature sets: X_train, X_val, X_test")

    # Save target variables
    y_train_path = output_dir / "y_train.csv"
    y_val_path = output_dir / "y_val.csv"
    y_test_path = output_dir / "y_test.csv"

    y_train.to_csv(y_train_path, index=False)
    y_val.to_csv(y_val_path, index=False)
    y_test.to_csv(y_test_path, index=False)

    logger.info(f"Saved target variables: y_train, y_val, y_test")

    # Log class distribution
    logger.info(f"\nClass Distribution:")
    logger.info(f"Train - Default: {(y_train == 1).sum()} | No Default: {(y_train == 0).sum()}")
    logger.info(f"Val   - Default: {(y_val == 1).sum()} | No Default: {(y_val == 0).sum()}")
    logger.info(f"Test  - Default: {(y_test == 1).sum()} | No Default: {(y_test == 0).sum()}")


@app.command()
def main(
    input_path: Path = typer.Option(
        RAW_DATA_DIR / "credit_risk_dataset.csv",
        "--input-path",
        "-i",
        help="Path to raw dataset CSV",
    ),
    output_dir: Path = typer.Option(
        INTERIM_DATA_DIR, "--output-dir", "-o", help="Directory to save interim data"
    ),
    test_size: float = typer.Option(
        0.3, "--test-size", "-t", help="Proportion for test+val split (default: 0.3 = 30%)"
    ),
    random_state: int = typer.Option(
        42, "--random-state", "-r", help="Random seed for reproducibility"
    ),
):
    """
    Load raw credit risk data and create train/validation/test splits.

    This is Stage 1 of the data pipeline. Performs basic preprocessing including:
    - Duplicate removal
    - Train/validation/test stratified split (70/15/15)
    - Saves preprocessed data and splits to interim directory

    Outputs:
    - credit_risk_dataset_prepped.csv: Full dataset after preprocessing
    - X_train.csv, X_val.csv, X_test.csv: Feature sets
    - y_train.csv, y_val.csv, y_test.csv: Target variables
    """
    logger.info("=" * 60)
    logger.info("STAGE 1: Raw Data → Interim Data")
    logger.info("=" * 60)

    try:
        # Load data
        df = load_raw_data(input_path)

        # Handle duplicates
        df = handle_duplicates(df)

        # Create splits
        X_train, X_val, X_test, y_train, y_val, y_test = create_splits(
            df, test_size=test_size, random_state=random_state
        )

        # Save outputs
        save_splits(df, X_train, X_val, X_test, y_train, y_val, y_test, output_dir)

        logger.success("=" * 60)
        logger.success("Stage 1 completed successfully!")
        logger.success("=" * 60)

    except Exception as e:
        logger.error(f"Stage 1 failed: {str(e)}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
