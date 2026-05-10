import numpy as np
import pandas as pd
import optuna
import joblib
from pathlib import Path
from typing import List, Tuple, Optional

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from src.models.walk_forward import (
    WalkForwardFold,
    WalkForwardConfig,
    generate_folds,
    get_feature_columns,
)
from src.utils.logger import log
from src.utils.exceptions import ModelTrainingError
from src.utils.config import CONFIG

# Suppress Optuna's default verbose logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


"""
Sklearn Model Zoo with Optuna Hyperparameter Tuning

Trains and tunes multiple sklearn-compatible classifiers.
Optuna searches across both model TYPE and hyperparameters
in a single unified search — it picks the best combination.

Models in the zoo:
- LightGBM    → Best for tabular data, fast, handles missing values
- XGBoost     → Strong baseline, well understood in finance
- RandomForest → Good uncertainty estimates, stable
- LogisticRegression → Interpretable, strong regularisation
- RidgeClassifier → Fast, good for linearly separable regimes

Evaluation metric: Mean Sharpe Ratio across all walk-forward folds
"""


# ── Sharpe Ratio Calculator ────────────────────────────────────────────────────

def compute_sharpe(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """
    Compute annualised Sharpe Ratio from daily strategy returns.

    Args:
        returns:        Array of daily strategy returns
        risk_free_rate: Annual risk-free rate (default 0 for simplicity)

    Returns:
        Annualised Sharpe Ratio (0.0 if computation fails)
    """
    if len(returns) < 2:
        return 0.0

    excess = returns - risk_free_rate / 252
    std    = np.std(excess)

    if std == 0:
        return 0.0

    return float(np.mean(excess) / std * np.sqrt(252))


def signal_to_returns(
    predictions: np.ndarray,
    actual_returns: np.ndarray,
) -> np.ndarray:
    """
    Convert binary predictions to strategy returns.

    Logic:
    - Prediction = 1 (BUY)  → go long  → return = +actual_return
    - Prediction = 0 (SELL) → go short → return = -actual_return

    Args:
        predictions:    Binary predictions (0 or 1)
        actual_returns: Actual daily log returns

    Returns:
        Strategy returns array
    """
    positions = np.where(predictions == 1, 1, -1)
    return positions * actual_returns


# ── Model Builder ──────────────────────────────────────────────────────────────

def build_model(trial: optuna.Trial) -> Pipeline:
    """
    Build a sklearn Pipeline based on Optuna's trial suggestions.
    Optuna picks the model type AND its hyperparameters together.

    Args:
        trial: Optuna trial object

    Returns:
        Sklearn Pipeline with StandardScaler + chosen classifier
    """

    model_type = trial.suggest_categorical(
        "model_type",
        ["lightgbm", "xgboost", "random_forest", "logistic", "ridge"]
    )

    if model_type == "lightgbm":
        clf = LGBMClassifier(
            n_estimators     = trial.suggest_int("lgbm_n_est", 100, 1000),
            learning_rate    = trial.suggest_float("lgbm_lr", 1e-3, 0.3, log=True),
            max_depth        = trial.suggest_int("lgbm_depth", 3, 10),
            num_leaves       = trial.suggest_int("lgbm_leaves", 20, 300),
            min_child_samples= trial.suggest_int("lgbm_min_child", 10, 100),
            subsample        = trial.suggest_float("lgbm_subsample", 0.5, 1.0),
            colsample_bytree = trial.suggest_float("lgbm_colsample", 0.5, 1.0),
            reg_alpha        = trial.suggest_float("lgbm_alpha", 1e-8, 10.0, log=True),
            reg_lambda       = trial.suggest_float("lgbm_lambda", 1e-8, 10.0, log=True),
            random_state     = 42,
            verbose          = -1,
        )

    elif model_type == "xgboost":
        clf = XGBClassifier(
            n_estimators  = trial.suggest_int("xgb_n_est", 100, 1000),
            learning_rate = trial.suggest_float("xgb_lr", 1e-3, 0.3, log=True),
            max_depth     = trial.suggest_int("xgb_depth", 3, 10),
            subsample     = trial.suggest_float("xgb_subsample", 0.5, 1.0),
            colsample_bytree = trial.suggest_float("xgb_colsample", 0.5, 1.0),
            reg_alpha     = trial.suggest_float("xgb_alpha", 1e-8, 10.0, log=True),
            reg_lambda    = trial.suggest_float("xgb_lambda", 1e-8, 10.0, log=True),
            random_state  = 42,
            eval_metric   = "logloss",
            verbosity     = 0,
        )

    elif model_type == "random_forest":
        clf = RandomForestClassifier(
            n_estimators = trial.suggest_int("rf_n_est", 100, 1000),
            max_depth    = trial.suggest_int("rf_depth", 3, 20),
            min_samples_split = trial.suggest_int("rf_min_split", 2, 20),
            min_samples_leaf  = trial.suggest_int("rf_min_leaf", 1, 10),
            max_features = trial.suggest_categorical("rf_features", ["sqrt", "log2"]),
            random_state = 42,
            n_jobs       = -1,
        )

    elif model_type == "logistic":
        clf = LogisticRegression(
            C        = trial.suggest_float("lr_C", 1e-4, 100.0, log=True),
            penalty  = trial.suggest_categorical("lr_penalty", ["l1", "l2"]),
            solver   = "liblinear",
            max_iter = 1000,
            random_state = 42,
        )

    else:  # ridge
        clf = RidgeClassifier(
            alpha = trial.suggest_float("ridge_alpha", 1e-4, 100.0, log=True),
        )

    # Always scale features — critical for logistic/ridge, harmless for trees
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", clf),
    ])


# ── Objective Function ─────────────────────────────────────────────────────────

def objective(
    trial: optuna.Trial,
    folds: List[WalkForwardFold],
    feature_cols: List[str],
    risk_free_rate: float,
) -> float:
    """
    Optuna objective function.
    Builds a model, evaluates it across ALL walk-forward folds,
    and returns the MEAN Sharpe Ratio.

    Averaging across folds prevents overfitting to any single
    market regime or val period.

    Args:
        trial:          Optuna trial
        folds:          List of WalkForwardFold objects
        feature_cols:   Feature column names
        risk_free_rate: Annual risk-free rate from config

    Returns:
        Mean Sharpe Ratio across all folds (higher = better)
    """

    model    = build_model(trial)
    sharpes  = []

    for fold in folds:
        try:
            # ── Prepare data ───────────────────────────────────────────────
            X_train = fold.train[feature_cols].values
            y_train = fold.train["target"].values

            # Val + test combined for evaluation
            X_eval  = pd.concat([fold.val, fold.test])
            y_eval  = X_eval["target"].values
            r_eval  = X_eval["log_return"].values
            X_eval  = X_eval[feature_cols].values

            # ── Train ──────────────────────────────────────────────────────
            model.fit(X_train, y_train)

            # ── Predict ────────────────────────────────────────────────────
            preds = model.predict(X_eval)

            # ── Evaluate ───────────────────────────────────────────────────
            strategy_returns = signal_to_returns(preds, r_eval)
            sharpe = compute_sharpe(strategy_returns, risk_free_rate)
            sharpes.append(sharpe)

        except Exception as e:
            # One bad fold doesn't kill the trial — penalise it
            log.warning(f"Trial {trial.number} fold {fold.fold_id} failed: {e}")
            sharpes.append(-1.0)

    mean_sharpe = float(np.mean(sharpes))

    log.debug(
        f"Trial {trial.number:>4} | "
        f"Model: {trial.params.get('model_type', 'unknown'):>15} | "
        f"Mean Sharpe: {mean_sharpe:.4f}"
    )

    return mean_sharpe


# ── Main Training Function ─────────────────────────────────────────────────────

def train_best_sklearn(
    df: pd.DataFrame,
    ticker: str,
    n_trials: int = 100,
    output_dir: str = "models/sklearn",
) -> Tuple[Pipeline, dict]:
    """
    Run full Optuna study to find the best sklearn model and params.
    Saves the winning model to disk.

    Args:
        df:         Full feature DataFrame for this ticker
        ticker:     Ticker name for logging and file naming
        n_trials:   Number of Optuna trials (more = better, slower)
        output_dir: Where to save the best model

    Returns:
        Tuple of (best_pipeline, study_results_dict)

    Raises:
        ModelTrainingError: If study fails or no valid trials complete
    """

    log.info("=" * 60)
    log.info(f"SKLEARN MODEL SELECTION — {ticker}")
    log.info(f"Trials: {n_trials}")
    log.info("=" * 60)

    # ── Setup walk-forward folds ───────────────────────────────────────────
    wf_config = WalkForwardConfig(
        train_window = CONFIG["backtesting"]["train_months"] * 21,
        val_window   = 63,
        test_window  = 21,
        step_size    = 21,
        holdout_pct  = 0.20,
    )

    feature_cols   = get_feature_columns(df)
    risk_free_rate = CONFIG["backtesting"]["risk_free_rate"]

    try:
        folds = generate_folds(df, wf_config)
    except Exception as e:
        raise ModelTrainingError(
            f"Failed to generate folds for {ticker}",
            details={"error": str(e)}
        )

    log.info(f"Running Optuna over {len(folds)} walk-forward folds...")

    # ── Create and run Optuna study ────────────────────────────────────────
    study = optuna.create_study(
        direction   = "maximize",          # Maximise Sharpe Ratio
        study_name  = f"sklearn_{ticker}",
        sampler     = optuna.samplers.TPESampler(seed=42),  # Bayesian
        pruner      = optuna.pruners.MedianPruner(),        # Kill bad trials early
    )

    try:
        study.optimize(
            lambda trial: objective(trial, folds, feature_cols, risk_free_rate),
            n_trials         = n_trials,
            show_progress_bar= True,
        )
    except Exception as e:
        raise ModelTrainingError(
            f"Optuna study failed for {ticker}",
            details={"error": str(e)}
        )

    # ── Extract best trial ─────────────────────────────────────────────────
    best_trial  = study.best_trial
    best_sharpe = best_trial.value
    best_params = best_trial.params
    best_model_type = best_params.get("model_type", "unknown")

    log.success(
        f"Best model: {best_model_type} | "
        f"Sharpe: {best_sharpe:.4f} | "
        f"Trial: {best_trial.number}"
    )
    log.info(f"Best params: {best_params}")

    # ── Retrain best model on ALL tuning data ──────────────────────────────
    # After finding best params, retrain on full tuning set
    # (not just the last fold) for maximum signal
    log.info("Retraining best model on full tuning dataset...")

    holdout_size  = int(len(df) * wf_config.holdout_pct)
    tuning_data   = df.iloc[:-holdout_size]

    X_full = tuning_data[feature_cols].values
    y_full = tuning_data["target"].values

    best_model = build_model(best_trial)
    best_model.fit(X_full, y_full)

    # ── Save model ─────────────────────────────────────────────────────────
    clean_ticker = ticker.replace("=X", "").replace("/", "")
    save_dir = Path(output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    model_path = save_dir / f"{clean_ticker}_sklearn_best.pkl"

    try:
        joblib.dump(best_model, model_path)
        log.success(f"Best sklearn model saved → {model_path}")
    except Exception as e:
        raise ModelTrainingError(
            f"Failed to save sklearn model for {ticker}",
            details={"path": str(model_path), "error": str(e)}
        )

    results = {
        "ticker":       ticker,
        "best_model":   best_model_type,
        "best_sharpe":  best_sharpe,
        "best_params":  best_params,
        "n_trials":     n_trials,
        "n_folds":      len(folds),
        "model_path":   str(model_path),
        "feature_cols": feature_cols,
    }

    return best_model, results
