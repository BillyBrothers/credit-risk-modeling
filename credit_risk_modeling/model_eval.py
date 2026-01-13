import numpy as np

def get_model_label(est, index=None, include_params=None):
    """ Pass an estimator and will return name of estimator and used hyperparameters"""
    name = est.__class__.__name__
    params = est.get_params()

    key_parts = []

    # Include parameters that distinguish your models
    if include_params is None:
        include_params = ['penalty', 'solver', 'n_jobs', 'l1_ratio', 'l1_ratios', 'Cs', 'class_weight']
    for p in include_params:
        if p in params:
            val = params[p]

            # Convert numpy arrays to lists for stable string formatting
            if isinstance(val, np.ndarray):
                val = val.tolist()

            key_parts.append(f"{p}={val}")

    label = f"{name} ({', '.join(key_parts)})" if key_parts else name

    # Add index suffix to guarantee uniqueness
    if index is not None:
        label = f"{label}__{index}"

    return label


def comparing_models(models, X_train, y_train, X_test, y_test):
    """ Pass a list of models and the function with calculate predictions and positive probabilites and use those to produce classification
     based performance metrics. A dataframe containing the performance metrics for each model and the fitted model will be stored in a dictionary. """
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

    return pd.DataFrame(results), fitted_models


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

    return pd.DataFrame(results), fitted_models_manual

def combine_model_performance(internal_tuned_models_df, manual_tuned_models_df):
    """Concatanate your performance dataframes."""
    return pd.concat(
    objs = [internal_tuned_models_df, manual_tuned_models_df],
    axis=0
    ).sort_values(
    by = 'roc_auc',
    ascending=False
    )

def evaluate_calibration(final_model_performances, preferred_fitted_models, X_train, y_train, X_test, y_test):
    """Final_model_performance parameter is the concatenation returned by combine_model_performance function. Fitted Models parameter is returned by your comparing models function and depends on dictionary of models you chose
    (e.g. internally tuned models or manually tuned models)"""
    top_model_name = final_model_performances.iloc[0]['model']
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
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
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

