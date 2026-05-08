import pandas as pd
import numpy as np
import optuna
import joblib
from pathlib import Path
from typing import List, Tuple
from prophet import Prophet

from src.models.walk_forward import (
    WalkForwardFold,
    WalkForwardConfig,
    generate_folds,
)
from src.models.sklearn_model import compute_sharpe, signal_to_returns
from src.utils.logger import log
from src.utils.exceptions import ModelTrainingError
from src.utils.config import CONFIG

optuna.logging.set_verbosity(optuna.logging.WARNING)

"""
Prophet Model with Optuna Tuning

Prophet is a decomposable time series model from Meta.
It handles trend, seasonality, and holiday effects well.
It is particularly useful for capturing weekly/monthly
cyclical patterns in forex markets.

In our ensemble:
- Prophet captures long-term trend and seasonality signals
- Its predictions are probability-like scores (0 to 1)
- These are combined with LSTM and sklearn outputs
"""


# ── Prophet Helpers ────────────────────────────────────────────────────────────

def prepare_prophet_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prophet requires a DataFrame with exactly two columns:
    'ds' (datestamp) and 'y' (value to forecast).

    We forecast the Close price, then convert to
    directional signals in post-processing.

    Args:
        df: Feature DataFrame with DatetimeIndex

    Returns:
        Prophet-formatted DataFrame
    """
    prophet_df = pd.DataFrame({
        "ds": df.index.tz_localize(None),  # Prophet doesn't handle tz-aware
        "y":  df["Close"].values,
    })
    return prophet_df


def get_prophet_signal(
    model: Prophet,
    forecast_df: pd.DataFrame,
    actual_df: pd.DataFrame,
    horizon: int,
) -> np.ndarray:
    """
    Convert Prophet forecast into binary directional signals.

    Logic: If forecasted price in `horizon` days > today's price → BUY (1)
           Otherwise → SELL (0)

    Args:
        model:       Fitted Prophet model
        forecast_df: DataFrame with future dates for prediction
        actual_df:   Actual data (for reference close prices)
        horizon:     Prediction horizon in days

    Returns:
        Binary signal array aligned with actual_df
    """

    forecast = model.predict(forecast_df)
    signals  = []

    for i in range(len(actual_df)):
        current_price   = actual_df["Close"].iloc[i]
        future_forecast = forecast["yhat"].iloc[i + horizon] \
            if (i + horizon) < len(forecast) else current_price

        signal = 1 if future_forecast > current_price else 0
        signals.append(signal)

    return np.array(signals)


# ── Objective Function ─────────────────────────────────────────────────────────

def prophet_objective(
    trial: optuna.Trial,
    folds: List[WalkForwardFold],
    horizon: int,
    risk_free_rate: float,
) -> float:
    """
    Optuna objective for Prophet hyperparameter tuning.

    Tuned parameters:
    - changepoint_prior_scale: Flexibility of trend changes
      (high = more flexible = risk of overfitting to noise)
    - seasonality_prior_scale: Strength of seasonality component
    - seasonality_mode:        Additive vs multiplicative seasonality
    - changepoint_range:       Fraction of history for trend changepoints

    Returns:
        Mean Sharpe Ratio across all walk-forward folds
    """

    params = {
        "changepoint_prior_scale":  trial.suggest_float(
            "changepoint_prior_scale", 0.001, 0.5, log=True
        ),
        "seasonality_prior_scale":  trial.suggest_float(
            "seasonality_prior_scale", 0.01, 10.0, log=True
        ),
        "seasonality_mode":         trial.suggest_categorical(
            "seasonality_mode", ["additive", "multiplicative"]
        ),
        "changepoint_range":        trial.suggest_float(
            "changepoint_range", 0.8, 0.95
        ),
        "yearly_seasonality":       trial.suggest_categorical(
            "yearly_seasonality", [True, False]
        ),
        "weekly_seasonality":       trial.suggest_categorical(
            "weekly_seasonality", [True, False]
        ),
    }

    sharpes = []

    for fold in folds:
        try:
            train_prophet = prepare_prophet_df(fold.train)
            eval_data     = pd.concat([fold.val, fold.test])

            # Build future dates for prediction
            future_dates = pd.DataFrame({
                "ds": pd.date_range(
                    start  = train_prophet["ds"].iloc[0],
                    periods= len(fold.train) + len(eval_data) + horizon,
                    freq   = "B",   # Business days
                )
            })

            # Fit Prophet — suppress stdout output
            model = Prophet(**params)

            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(train_prophet)

            signals = get_prophet_signal(
                model       = model,
                forecast_df = future_dates,
                actual_df   = eval_data,
                horizon     = horizon,
            )

            actual_returns   = eval_data["log_return"].values[:len(signals)]
            strategy_returns = signal_to_returns(signals, actual_returns)
            sharpe           = compute_sharpe(strategy_returns, risk_free_rate)
            sharpes.append(sharpe)

        except Exception as e:
            log.warning(f"Prophet trial {trial.number} fold {fold.fold_id} failed: {e}")
            sharpes.append(-1.0)

    mean_sharpe = float(np.mean(sharpes))
    log.debug(f"Prophet Trial {trial.number:>4} | Mean Sharpe: {mean_sharpe:.4f}")
    return mean_sharpe


# ── Main Training Function ─────────────────────────────────────────────────────

def train_prophet(
    df: pd.DataFrame,
    ticker: str,
    n_trials: int = 50,
    output_dir: str = "models/prophet",
) -> Tuple[Prophet, dict]:
    """
    Tune and train the Prophet model using Optuna + walk-forward validation.

    Args:
        df:         Feature DataFrame
        ticker:     Ticker symbol
        n_trials:   Number of Optuna trials
        output_dir: Where to save the fitted model

    Returns:
        Tuple of (fitted_prophet_model, results_dict)

    Raises:
        ModelTrainingError: If training fails
    """

    log.info("=" * 60)
    log.info(f"PROPHET MODEL TRAINING — {ticker}")
    log.info(f"Trials: {n_trials}")
    log.info("=" * 60)

    wf_config = WalkForwardConfig(
        train_window = CONFIG["backtesting"]["train_months"] * 21,
        val_window   = 63,
        test_window  = 21,
        step_size    = 21,
        holdout_pct  = 0.20,
    )

    horizon        = CONFIG["features"]["prediction_horizon"]
    risk_free_rate = CONFIG["backtesting"]["risk_free_rate"]

    try:
        folds = generate_folds(df, wf_config)
    except Exception as e:
        raise ModelTrainingError(
            f"Failed to generate folds for Prophet — {ticker}",
            details={"error": str(e)}
        )

    # ── Optuna study ───────────────────────────────────────────────────────
    study = optuna.create_study(
        direction  = "maximize",
        study_name = f"prophet_{ticker}",
        sampler    = optuna.samplers.TPESampler(seed=42),
    )

    study.optimize(
        lambda trial: prophet_objective(trial, folds, horizon, risk_free_rate),
        n_trials          = n_trials,
        show_progress_bar = True,
    )

    best_params = study.best_trial.params
    best_sharpe = study.best_trial.value

    log.success(f"Best Prophet Sharpe: {best_sharpe:.4f}")
    log.info(f"Best params: {best_params}")

    # ── Retrain on full tuning data ────────────────────────────────────────
    holdout_size = int(len(df) * wf_config.holdout_pct)
    tuning_data  = df.iloc[:-holdout_size]
    train_df     = prepare_prophet_df(tuning_data)

    final_model = Prophet(**best_params)

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final_model.fit(train_df)

    # ── Save model ─────────────────────────────────────────────────────────
    clean_ticker = ticker.replace("=X", "").replace("/", "")
    save_dir     = Path(output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    model_path   = save_dir / f"{clean_ticker}_prophet.pkl"

    joblib.dump(final_model, model_path)
    log.success(f"Prophet model saved → {model_path}")

    results = {
        "ticker":      ticker,
        "best_sharpe": best_sharpe,
        "best_params": best_params,
        "n_trials":    n_trials,
        "n_folds":     len(folds),
        "model_path":  str(model_path),
    }

    return final_model, results
EOF