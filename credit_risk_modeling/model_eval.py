# data analysis
import pandas as pd
import numpy as np

# visualization
import matplotlib.pyplot as plt
import seaborn as sns
from pyampute.exploration.md_patterns import mdPatterns
from pyampute.exploration.mcar_statistical_tests import MCARTest
import missingno as msno


# preprocessing
import sklearn.utils.validation
import sys
from scipy import stats
from scipy.stats import shapiro, distributions, loguniform
from scipy.stats.mstats import winsorize
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, GridSearchCV, HalvingRandomSearchCV, HalvingGridSearchCV, TunedThresholdClassifierCV, FixedThresholdClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, PowerTransformer, QuantileTransformer, MinMaxScaler, KBinsDiscretizer, Binarizer, PolynomialFeatures, LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer, make_column_selector
from feature_engine.outliers import Winsorizer
# from imblearn.over_sampling import SMOTE, SMOTENC
from sklearn.pipeline import Pipeline
# from sklearn import set_config

# Feature Selection
from sklearn.feature_selection import SelectFromModel

# Modeling
from sklearn.linear_model import RidgeClassifier, LogisticRegression, RidgeClassifierCV, LogisticRegressionCV, SGDClassifier, Perceptron, PassiveAggressiveClassifier
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
import joblib
from sklearn import tree 
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier, VotingClassifier, StackingClassifier
from lightgbm import LGBMClassifier 

# Metrics
from sklearn.metrics import confusion_matrix, recall_score, precision_score, balanced_accuracy_score, ConfusionMatrixDisplay, classification_report, precision_recall_curve, PrecisionRecallDisplay, log_loss, brier_score_loss, roc_curve, roc_auc_score, RocCurveDisplay, det_curve, DetCurveDisplay, fbeta_score, average_precision_score, matthews_corrcoef

# Calibration
from sklearn.calibration import calibration_curve, CalibrationDisplay, CalibratedClassifierCV

# Inspection
from sklearn.inspection import PartialDependenceDisplay

# Custom Functions
from credit_risk_modeling import model_eval
import importlib
importlib.reload(model_eval)

import tensorflow as tf
from tensorflow import keras
from keras import layers
from scikeras.wrappers import KerasClassifier
import keras_tuner as kt

# Class Imbalance
from imblearn.over_sampling import SMOTE

def get_model_label(est, index=None, include_params=None):
    """Return a readable label for an estimator, including SciKeras models."""

    # Special handling for SciKeras KerasClassifier
    if hasattr(est, "model"):
        # est.model is a Sequential instance, so use its .name attribute
        try:
            name = est.model.name
        except AttributeError:
            name = est.__class__.__name__
    else:
        name = est.__class__.__name__

    params = est.get_params()
    key_parts = []

    if include_params is None:
        include_params = ['penalty', 'solver', 'n_jobs', 'l1_ratio', 'l1_ratios', 'Cs']

    for p in include_params:
        if p in params:
            val = params[p]
            if isinstance(val, np.ndarray):
                val = val.tolist()
            key_parts.append(f"{p}={val}")

    label = f"{name} ({', '.join(key_parts)})" if key_parts else name

    if index is not None:
        label = f"{label}__{index}"

    return label


def comparing_models(models, X_train, y_train, X_test, y_test):
    """ Pass a list of models and the function with calculate predictions and positive probabilites and use those to produce classification
     based performance metrics. A dataframe containing the performance metrics for each model and the fitted model will be stored in a dictionary. used to 
      compare both untuned models and internally tuned models. For manually tuned models use comparing_manually_tuned_models. """
    results = []
    fitted_models = {}

    for est in models:
        model_name = get_model_label(est)

        try:
            est.fit(X_train, y_train)
            fitted_models[model_name] = est 

            y_pred = est.predict(X_test)

            roc_auc = np.nan
            pr_auc = np.nan
            ll = np.nan
            bcl = np.nan

            if hasattr(est, "predict_proba"):
                pos_prob = est.predict_proba(X_test)[:, 1]
                roc_auc = roc_auc_score(y_test, pos_prob)
                pr_auc = average_precision_score(y_test, pos_prob)
                ll = log_loss(y_test, pos_prob)
                bcl = brier_score_loss(y_test, pos_prob)

            elif hasattr(est, "decision_function"):
                scores = est.decision_function(X_test)
                roc_auc = roc_auc_score(y_test, scores)
                pr_auc = average_precision_score(y_test, scores)

            mc = matthews_corrcoef(y_test, y_pred)

            results.append({
                "model": model_name,
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
                "log_loss": ll,
                "brier_score": bcl,
                "matthews_corrcoef": mc,
            })

        except Exception as e:
            results.append({
                "model": model_name,
                "roc_auc": np.nan,
                "pr_auc": np.nan,
                "log_loss": np.nan,
                "brier_score": np.nan,
                "matthews_corrcoef": np.nan,
                "error": str(e)
            })
    df = pd.DataFrame(results).sort_values(
        ascending=False,
        by= 'roc_auc'
    )

    return df, fitted_models


def comparing_models_smoted(models, X_train, y_train, X_test, y_test, sampling_strategy='minority'):
    """ Pass a list of models and the function with calculate predictions and positive probabilites and use those to produce classification
     based performance metrics. A dataframe containing the performance metrics for each model and the fitted model will be stored in a dictionary. Used to 
      compare both untuned models and internally tuned models. For manually tuned models use comparing_manually_tuned_models. Used for models that don't internally handle. """
    results = []
    fitted_models = {}

    for est in models:
        model_name = get_model_label(est)

        try:
            smote = SMOTE(sampling_strategy=sampling_strategy)
            X_trained_smoted, y_train_smoted = smote.fit_resample(X_train, y_train)

            est.fit(X_trained_smoted, y_train_smoted)
            fitted_models[model_name] = est 

            y_pred = est.predict(X_test)

            roc_auc = np.nan
            pr_auc = np.nan
            ll = np.nan
            bcl = np.nan

            if hasattr(est, "predict_proba"):
                pos_prob = est.predict_proba(X_test)[:, 1]
                roc_auc = roc_auc_score(y_test, pos_prob)
                pr_auc = average_precision_score(y_test, pos_prob)
                ll = log_loss(y_test, pos_prob)
                bcl = brier_score_loss(y_test, pos_prob)

            elif hasattr(est, "decision_function"):
                scores = est.decision_function(X_test)
                roc_auc = roc_auc_score(y_test, scores)
                pr_auc = average_precision_score(y_test, scores)

            mc = matthews_corrcoef(y_test, y_pred)

            results.append({
                "model": model_name,
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
                "log_loss": ll,
                "brier_score": bcl,
                "matthews_corrcoef": mc,
            })

        except Exception as e:
            results.append({
                "model": model_name,
                "roc_auc": np.nan,
                "pr_auc": np.nan,
                "log_loss": np.nan,
                "brier_score": np.nan,
                "matthews_corrcoef": np.nan,
                "error": str(e)
            })
    df = pd.DataFrame(results).sort_values(
        ascending=False,
        by= 'roc_auc'
    )

    return df, fitted_models


def comparing_manually_tuned_models(models, X_train, y_train, X_test, y_test):
    """Pass a list of models that required prior manual hyperparameter tuning and are wrapped in a tuning object (e.g. RandomizedSearchCV) and the function with calculate predictions and positive probabilites and use those to produce classification
     based performance metrics. A dataframe containing the performance metrics for each model and the fitted model will be stored in a dictionary. """
    results = []
    fitted_models_manual = {}

    for est in models:
        model_name = get_model_label(est.estimator)

        try:
            est.fit(X_train, y_train)
            fitted_models_manual[model_name] = est.best_estimator_ 

            y_pred = est.best_estimator_.predict(X_test)

            roc_auc = np.nan
            pr_auc = np.nan
            ll = np.nan
            bcl = np.nan

            if hasattr(est.best_estimator_, "predict_proba"):
                pos_prob = est.best_estimator_.predict_proba(X_test)[:, 1]
                roc_auc = roc_auc_score(y_test, pos_prob)
                pr_auc = average_precision_score(y_test, pos_prob)
                ll = log_loss(y_test, pos_prob)
                bcl = brier_score_loss(y_test, pos_prob)

            elif hasattr(est.best_estimator_, "decision_function"):
                scores = est.best_estimator_.decision_function(X_test)
                roc_auc = roc_auc_score(y_test, scores)
                pr_auc = average_precision_score(y_test, scores)

            mc = matthews_corrcoef(y_test, y_pred)

            results.append({
                "model": model_name,
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
                "log_loss": ll,
                "brier_score": bcl,
                "matthews_corrcoef": mc
            })

        except Exception as e:
            results.append({
                "model": model_name,
                "roc_auc": np.nan,
                "pr_auc": np.nan,
                "log_loss": np.nan,
                "brier_score": np.nan,
                "matthews_corrcoef": np.nan,
                "error": str(e)
            })

        df= pd.DataFrame(results).sort_values(
            ascending=False,
            by= 'roc_auc'
        )
    return df, fitted_models_manual

def comparing_manually_tuned_smoted_models(models, X_train, y_train, X_test, y_test, sampling_strategy='minority'):
    """Pass a list of models that required prior manual hyperparameter tuning and are wrapped in a tuning object (e.g. RandomizedSearchCV) and the function with calculate predictions and positive probabilites and use those to produce classification
     based performance metrics. A dataframe containing the performance metrics for each model and the fitted model will be stored in a dictionary. """
    results = []
    fitted_models_manual = {}

    for est in models:
        model_name = get_model_label(est.estimator)

        try:
            smote = SMOTE(sampling_strategy=sampling_strategy)
            X_trained_smoted, y_train_smoted = smote.fit_resample(X_train, y_train)
            est.fit(X_trained_smoted, y_train_smoted)
            fitted_models_manual[model_name] = est.best_estimator_ 

            y_pred = est.best_estimator_.predict(X_test)

            roc_auc = np.nan
            pr_auc = np.nan
            ll = np.nan
            bcl = np.nan

            if hasattr(est.best_estimator_, "predict_proba"):
                pos_prob = est.best_estimator_.predict_proba(X_test)[:, 1]
                roc_auc = roc_auc_score(y_test, pos_prob)
                pr_auc = average_precision_score(y_test, pos_prob)
                ll = log_loss(y_test, pos_prob)
                bcl = brier_score_loss(y_test, pos_prob)

            elif hasattr(est.best_estimator_, "decision_function"):
                scores = est.best_estimator_.decision_function(X_test)
                roc_auc = roc_auc_score(y_test, scores)
                pr_auc = average_precision_score(y_test, scores)

            mc = matthews_corrcoef(y_test, y_pred)

            results.append({
                "model": model_name,
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
                "log_loss": ll,
                "brier_score": bcl,
                "matthews_corrcoef": mc
            })

        except Exception as e:
            results.append({
                "model": model_name,
                "roc_auc": np.nan,
                "pr_auc": np.nan,
                "log_loss": np.nan,
                "brier_score": np.nan,
                "matthews_corrcoef": np.nan,
                "error": str(e)
            })

        df= pd.DataFrame(results).sort_values(
            ascending=False,
            by= 'roc_auc'
        )
    return df, fitted_models_manual

def combine_model_performance(internal_tuned_models_df, manual_tuned_models_df, untuned_model_performance):
    """Concatanate your performance dataframes."""
    return pd.concat(
    objs = [internal_tuned_models_df, manual_tuned_models_df, untuned_model_performance],
    axis=0
    ).sort_values(
        ascending=False,
        by='roc_auc'
    )


def evaluate_calibration(final_model_performances, preferred_fitted_models, X_train, y_train, X_test, y_test):
    """Final_model_performance parameter is the concatenation returned by combine_model_performance function. Fitted Models parameter is returned by your comparing models function and depends on dictionary of models you chose
    (e.g. internally tuned models or manually tuned models)"""
    top_model_name = final_model_performances.iloc[0]['model']
    
    if hasattr(top_model_name, 'layers'):
            top_performing_model = KerasClassifier(
                                    model= top_model_name,
                                    optimizer= keras.optimizers.Adam(),
                                    loss= keras.losses.BinaryCrossentropy(),
                                    random_state=42,
                                    metrics= ['val_auc'],
                                    callbacks= [keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=1e-4,patience=7,verbose=1,restore_best_weights=True ),keras.callbacks.ReduceLROnPlateau(monitor="val_loss",factor=0.2,patience=3,verbose=1,min_lr=0.001)],
                                    validation_split= 0.20,
                                    epochs=100
                                )
    else:
            top_performing_model = preferred_fitted_models[top_model_name]

    methods = ['sigmoid', 'isotonic']


    calibrated_models = {}   # store models here

    fig, ax = plt.subplots(figsize=(8,6))

    for method in methods:
        # Fit calibrated model
        calibrated_model = CalibratedClassifierCV(
            estimator=top_performing_model,
            method=method,
            n_jobs=-1
        ).fit(X_train, y_train)

        # Store it for later use
        calibrated_models[method] = calibrated_model

        # Probabilities
        y_probs = calibrated_model.predict_proba(X_test)[:, 1]

        # Metrics
        ll = log_loss(y_test, y_probs)
        bcl = brier_score_loss(y_test, y_probs)

        print(f"Log loss ({method}): {ll:.4f}")
        print(f"Brier score ({method}): {bcl:.4f}")

        # Calibration curve
        CalibrationDisplay.from_predictions(
            y_true=y_test,
            y_prob=y_probs,
            n_bins=10,
            name=f"{method} calibration",
            ax=ax
        )

    # Perfect calibration line
    plt.title("Calibration Curves: Sigmoid vs Isotonic")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return calibrated_models 

def decision_threshold_comparisons(base_estimator, X_train, y_train, X_test, y_test,
                                   scoring='recall', fn_cost=10000, fp_cost=500, beta=2):
    
    """Evaluate Decision Thresholds for a base estimator. Returns estimators, classification reports, business decision cost (minimize false negatives) for base estimator, tuned estimator, optimally tuned estimator """

    fig, axes = plt.subplots(2, 3, figsize=(18,10))

    # --- Base Estimator ---
    base_pred = base_estimator.predict(X_test)
    base_cr = classification_report(y_true=y_test, y_pred=base_pred)
    print("Classification Report (Base Estimator):\n", base_cr)
    base_tn, base_fp, base_fn, base_tp = confusion_matrix(y_test, base_pred).ravel().tolist()
    print(f"Business Cost (Base Estimator): ${base_fn * fn_cost + base_fp * fp_cost}")

    ConfusionMatrixDisplay.from_predictions(y_test, base_pred, ax=axes[0,0], cmap='Blues')
    axes[0,0].set_title("Base Estimator")

    # --- Tuned ThresholdClassifier ---
    tuned_base_estimator = TunedThresholdClassifierCV(
        estimator=base_estimator,
        scoring=scoring,
        cv=5,
        n_jobs=-1,
        random_state=42
    ).fit(X_train, y_train)

    tuned_base_preds = tuned_base_estimator.predict(X_test)
    tuned_base_cr = classification_report(y_true=y_test, y_pred=tuned_base_preds)
    print("Classification Report (Tuned Estimator):\n", tuned_base_cr)
    tuned_base_tn, tuned_base_fp, tuned_base_fn, tuned_base_tp = confusion_matrix(y_test, tuned_base_preds).ravel().tolist()
    print(f"Business Cost (Tuned Estimator): ${tuned_base_fn * fn_cost + tuned_base_fp * fp_cost}")

    ConfusionMatrixDisplay.from_predictions(y_test, tuned_base_preds, ax=axes[0,1], cmap='Blues')
    axes[0,1].set_title("Tuned Estimator")

    # --- Step 3: Threshold Exploration ---
    # Safe probability extraction
    if hasattr(base_estimator, "predict_proba"):
        y_probs = base_estimator.predict_proba(X_test)[:, 1]

    elif hasattr(base_estimator, "decision_function"):
        scores = base_estimator.decision_function(X_test)
        y_probs = (scores - scores.min()) / (scores.max() - scores.min())

    else:
        raise ValueError(
            "The base estimator does not support predict_proba or decision_function, "
            "so threshold analysis cannot be performed."
        )

    thresholds = np.linspace(0.01, 0.99, 50)
    recalls, precisions, fbetas, losses = [], [], [], []

    for t in thresholds:
        preds = (y_probs >= t).astype(int)
        recalls.append(recall_score(y_test, preds))
        precisions.append(precision_score(y_test, preds))
        fbetas.append(fbeta_score(y_test, preds, beta=beta))
        tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
        losses.append(fn*fn_cost + fp*fp_cost)

    # Precision–Recall curve
    PrecisionRecallDisplay.from_predictions(y_true=y_test, y_pred=y_probs, ax=axes[1,0])

    # Expected Loss vs Threshold
    axes[1,1].plot(thresholds, losses, label="Expected Loss")
    axes[1,1].set_xlabel("Threshold")
    axes[1,1].set_ylabel("Expected Loss")
    axes[1,1].set_title("Expected Loss vs Threshold")
    best_idx = np.argmin(losses)
    best_threshold = thresholds[best_idx]
    axes[1,1].axvline(best_threshold, color='red', linestyle='--', label=f"Best={best_threshold:.2f}")
    axes[1,1].legend()

    # Chosen Threshold Model
    chosen_threshold_estimator = FixedThresholdClassifier(
        estimator=base_estimator,
        threshold=best_threshold
    ).fit(X_train, y_train)

    chosen_probs = chosen_threshold_estimator.predict_proba(X_test)[:, 1]
    chosen_preds = (chosen_probs >= best_threshold).astype(int)
    chosen_cr = classification_report(y_test, chosen_preds)
    print("Classification Report (Chosen Threshold):\n", chosen_cr)
    chosen_tn, chosen_fp, chosen_fn, chosen_tp = confusion_matrix(y_test, chosen_preds).ravel()
    print(f"Business Cost (Chosen Threshold): ${chosen_fn*fn_cost + chosen_fp*fp_cost}")

    ConfusionMatrixDisplay.from_predictions(y_test, chosen_preds, ax=axes[0,2], cmap='Blues')
    axes[0,2].set_title(f"Confusion Matrix @ {best_threshold:.2f}")

    plt.tight_layout()
    plt.show()

    return {
        "base_estimator": base_estimator,
        "tuned_estimator": tuned_base_estimator,
        "chosen_estimator": chosen_threshold_estimator,
        "base_report": base_cr,
        "tuned_report": tuned_base_cr,
        "chosen_report": chosen_cr,
        "base_cost": base_fn * fn_cost + base_fp * fp_cost,
        "tuned_cost": tuned_base_fn * fn_cost + tuned_base_fp * fp_cost,
        "chosen_cost": chosen_fn*fn_cost + chosen_fp*fp_cost,
        "best_threshold": best_threshold,
        "expected_loss_curve": list(zip(thresholds, losses)),
        "f_beta_curve": list(zip(thresholds, fbetas))
    }

