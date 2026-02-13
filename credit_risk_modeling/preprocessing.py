"""
Preprocessing module for credit risk modeling.

Handles data preprocessing including missing value imputation, outlier treatment,
transformations, encoding, and pipeline creation for different model types.
"""

from pathlib import Path
from typing import Tuple

import joblib
import pandas as pd
import typer
from loguru import logger
from sklearn.compose import ColumnTransformer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    PowerTransformer,
    StandardScaler,
    KBinsDiscretizer,
    PolynomialFeatures,
)
from feature_engine.outliers import Winsorizer

from credit_risk_modeling.config import INTERIM_DATA_DIR, MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


def load_interim_data(
    input_path: Path = INTERIM_DATA_DIR / "credit_risk_dataset_prepped.csv",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Load interim data and train/test/val splits."""
    logger.info(f"Loading data from {input_path}")
    df = pd.read_csv(input_path)

    # Load target variables
    y_train = pd.read_csv(INTERIM_DATA_DIR / "y_train.csv")
    y_test = pd.read_csv(INTERIM_DATA_DIR / "y_test.csv")
    y_val = pd.read_csv(INTERIM_DATA_DIR / "y_val.csv")

    # Get corresponding X splits
    X_indices_train = y_train.index if hasattr(y_train, "index") else range(len(y_train))
    X_indices_test = y_test.index if hasattr(y_test, "index") else range(len(y_test))
    X_indices_val = y_val.index if hasattr(y_val, "index") else range(len(y_val))

    # For now, load from previously saved files if they exist, or reconstruct
    X = df.drop("loan_status", axis=1, errors="ignore")
    y = df.get("loan_status", pd.Series())

    logger.success(f"Loaded data: {df.shape}")
    return X, y, df, y_train, y_test, y_val


def get_feature_columns(X: pd.DataFrame) -> Tuple[list, list]:
    """Identify numeric and categorical features."""
    numeric_features = list(X.select_dtypes(include=["int64", "float64", "bool"]).columns)
    categorical_features = list(X.select_dtypes(include=["object"]).columns)

    # Remove temporary flags if present
    for col in ["pel_missing", "lir_missing"]:
        if col in categorical_features:
            categorical_features.remove(col)

    return numeric_features, categorical_features


def build_numeric_pipeline(include_winsorizer: bool = False) -> Pipeline:
    """Build numeric features preprocessing pipeline."""
    steps = [
        ("imputer", IterativeImputer(max_iter=10, random_state=42)),
    ]

    if include_winsorizer:
        steps.append(
            ("winsorizer", Winsorizer(capping_method="iqr", tail="both", fold=1.5))
        )

    steps.append(("power_transform", PowerTransformer(method="yeo-johnson", standardize=False)))

    return Pipeline(steps=steps)


def build_ordinal_pipeline() -> Pipeline:
    """Build ordinal features preprocessing pipeline."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]
    )


def build_categorical_pipeline() -> Pipeline:
    """Build categorical features preprocessing pipeline."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", drop="if_binary", sparse_output=False)),
        ]
    )


def build_linear_pipeline(
    numeric_features: list, categorical_features: list
) -> Pipeline:
    """Build preprocessing pipeline for linear models."""
    numeric_pipe = build_numeric_pipeline(include_winsorizer=False)
    ordinal_pipe = build_ordinal_pipeline()
    categorical_pipe = build_categorical_pipeline()

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipe, numeric_features),
            ("ordinal", ordinal_pipe, ["loan_grade"]),
            ("categorical", categorical_pipe, categorical_features),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("polynomial", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
            ("scaling", StandardScaler()),
        ]
    )


def build_tree_pipeline(numeric_features: list, categorical_features: list) -> Pipeline:
    """Build preprocessing pipeline for tree-based models."""
    numeric_pipe = build_numeric_pipeline(include_winsorizer=True)
    ordinal_pipe = build_ordinal_pipeline()
    categorical_pipe = build_categorical_pipeline()

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipe, numeric_features),
            ("ordinal", ordinal_pipe, ["loan_grade"]),
            ("categorical", categorical_pipe, categorical_features),
        ]
    )

    return Pipeline(steps=[("preprocess", preprocessor)])


def build_distance_pipeline(numeric_features: list, categorical_features: list) -> Pipeline:
    """Build preprocessing pipeline for distance-based models."""
    numeric_pipe = build_numeric_pipeline(include_winsorizer=False)
    ordinal_pipe = build_ordinal_pipeline()
    categorical_pipe = build_categorical_pipeline()

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipe, numeric_features),
            ("ordinal", ordinal_pipe, ["loan_grade"]),
            ("categorical", categorical_pipe, categorical_features),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("scaling", StandardScaler()),
        ]
    )


def build_probability_pipeline(numeric_features: list, categorical_features: list) -> Pipeline:
    """Build preprocessing pipeline for probability-based models."""
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", IterativeImputer(max_iter=10, random_state=42)),
            ("power_transform", PowerTransformer(method="yeo-johnson", standardize=False)),
            ("kbins", KBinsDiscretizer(n_bins=3, encode="ordinal", strategy="quantile")),
            ("bin_encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first")),
        ]
    )
    ordinal_pipe = build_ordinal_pipeline()
    categorical_pipe = build_categorical_pipeline()

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipe, numeric_features),
            ("ordinal", ordinal_pipe, ["loan_grade"]),
            ("categorical", categorical_pipe, categorical_features),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("scaling", StandardScaler()),
        ]
    )


def build_neural_pipeline(numeric_features: list, categorical_features: list) -> Pipeline:
    """Build preprocessing pipeline for neural network models."""
    numeric_pipe = build_numeric_pipeline(include_winsorizer=False)
    ordinal_pipe = build_ordinal_pipeline()
    categorical_pipe = build_categorical_pipeline()

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipe, numeric_features),
            ("ordinal", ordinal_pipe, ["loan_grade"]),
            ("categorical", categorical_pipe, categorical_features),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("scaling", StandardScaler()),
        ]
    )


def fit_and_save_pipelines(
    X_train: pd.DataFrame, X_test: pd.DataFrame, X_val: pd.DataFrame, y_train: pd.Series = None
) -> None:
    """Fit all preprocessing pipelines and save them."""
    numeric_features, categorical_features = get_feature_columns(X_train)

    # Build pipelines
    pipelines = {
        "linear": build_linear_pipeline(numeric_features, categorical_features),
        "tree": build_tree_pipeline(numeric_features, categorical_features),
        "distance": build_distance_pipeline(numeric_features, categorical_features),
        "probability": build_probability_pipeline(numeric_features, categorical_features),
        "neural": build_neural_pipeline(numeric_features, categorical_features),
    }

    # Fit and transform
    transformed_data = {}

    for model_type, pipeline in pipelines.items():
        logger.info(f"Fitting {model_type} pipeline...")

        # Fit on training data
        X_train_transformed = pipeline.fit_transform(X_train, y_train)
        X_test_transformed = pipeline.transform(X_test)
        X_val_transformed = pipeline.transform(X_val)

        # Save pipeline
        pipeline_path = MODELS_DIR / f"{model_type}_preprocessed_pipeline.pkl"
        joblib.dump(pipeline, pipeline_path)
        logger.info(f"Saved pipeline to {pipeline_path}")

        # Save transformed data
        pd.DataFrame(X_train_transformed).to_csv(
            PROCESSED_DATA_DIR / f"X_train_{model_type}.csv", index=False
        )
        pd.DataFrame(X_test_transformed).to_csv(
            PROCESSED_DATA_DIR / f"X_test_{model_type}.csv", index=False
        )
        pd.DataFrame(X_val_transformed).to_csv(
            PROCESSED_DATA_DIR / f"X_val_{model_type}.csv", index=False
        )

        transformed_data[model_type] = {
            "train_shape": X_train_transformed.shape,
            "test_shape": X_test_transformed.shape,
            "val_shape": X_val_transformed.shape,
        }

        logger.success(f"Completed {model_type} pipeline")

    # Summary
    logger.info("\nPreprocessing Summary:")
    for model_type, shapes in transformed_data.items():
        logger.info(f"{model_type}: train={shapes['train_shape']}, test={shapes['test_shape']}, val={shapes['val_shape']}")


@app.command()
def main(
    input_path: Path = typer.Option(
        INTERIM_DATA_DIR / "credit_risk_dataset_prepped.csv",
        "--input-path",
        "-i",
        help="Path to interim preprocessed dataset",
    ),
    output_dir: Path = typer.Option(
        PROCESSED_DATA_DIR, "--output-dir", "-o", help="Directory to save processed data"
    ),
) -> None:
    """
    Preprocess credit risk data and create model-specific feature sets.

    Creates separate preprocessed datasets for linear, tree, distance, probability,
    and neural network models with appropriate transformations and encodings.
    """
    logger.info("Starting preprocessing pipeline...")

    try:
        # Load data
        X, y, df, y_train, y_test, y_val = load_interim_data(input_path)

        # Remove temporary flags
        for col in ["pel_missing", "lir_missing"]:
            if col in X.columns:
                X = X.drop(col, axis=1)

        # Split data (reconstruct from indices if needed)
        # For now, we'll assume X, X_test, X_val are properly aligned
        X_train = X.iloc[: len(y_train)]
        X_test = X.iloc[len(y_train) : len(y_train) + len(y_test)]
        X_val = X.iloc[len(y_train) + len(y_test) :]

        # Fit and save all pipelines
        fit_and_save_pipelines(X_train, X_test, X_val, y_train.iloc[:, 0] if len(y_train.shape) > 1 else y_train)

        logger.success("Preprocessing completed successfully!")

    except Exception as e:
        logger.error(f"Preprocessing failed: {str(e)}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
