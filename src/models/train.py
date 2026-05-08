import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from pathlib import Path

from src.models.sklearn_model import train_best_sklearn, signal_to_returns, compute_sharpe
from src.models.prophet_model import train_prophet, prepare_prophet_df, get_prophet_signal
from src.models.lstm_model import train_lstm, build_sequences
from src.models.ensemble import EnsemblePredictor, tune_ensemble_weights
from src.models.walk_forward import WalkForwardConfig, get_holdout, get_feature_columns
from src.utils.logger import log
from src.utils.exceptions import ModelTrainingError
from src.utils.config import CONFIG



"""
Master Training Entry Point

Orchestrates the full training pipeline for all tickers:
1. Load feature data
2. Train and tune sklearn zoo → select best model
3. Train and tune Prophet
4. Train and tune LSTM
5. Tune ensemble weights
6. Save ensemble predictor
7. Log everything to MLflow

Run this script to train all models:
    python -m src.models.train
"""


def train_ticker(ticker: str) -> dict:
    """
    Run the full training pipeline for a single forex pair.

    Args:
        ticker: yfinance ticker symbol e.g. 'EURUSD=X'

    Returns:
        Results dictionary with all metrics and paths
    """

    clean_ticker = ticker.replace("=X", "").replace("/", "")
    features_path = (
        Path(CONFIG["data"]["features_path"]) /
        f"{clean_ticker}_features.parquet"
    )

    # ── Load features ──────────────────────────────────────────────────────
    if not features_path.exists():
        raise ModelTrainingError(
            f"Feature file not found for {ticker}",
            details={"expected": str(features_path)}
        )

    df = pd.read_parquet(features_path)
    log.info(f"Loaded features for {ticker} | Shape: {df.shape}")

    feature_cols   = get_feature_columns(df)
    risk_free_rate = CONFIG["backtesting"]["risk_free_rate"]

    wf_config = WalkForwardConfig(
        train_window = CONFIG["backtesting"]["train_months"] * 21,
        val_window   = 63,
        test_window  = 21,
        step_size    = 21,
        holdout_pct  = 0.20,
    )

    # ── MLflow experiment ──────────────────────────────────────────────────
    mlflow.set_tracking_uri(CONFIG["mlflow"]["tracking_uri"])
    mlflow.set_experiment(CONFIG["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name=f"{clean_ticker}_training"):

        mlflow.log_param("ticker",         ticker)
        mlflow.log_param("n_features",     len(feature_cols))
        mlflow.log_param("train_rows",     len(df))
        mlflow.log_param("horizon_days",   CONFIG["features"]["prediction_horizon"])
        mlflow.log_param("risk_free_rate", risk_free_rate)

        # ── Train Sklearn ──────────────────────────────────────────────────
        log.info(f"[{ticker}] Training sklearn model zoo...")
        sklearn_model, sklearn_results = train_best_sklearn(
            df=df, ticker=ticker, n_trials=100
        )

        mlflow.log_metric("sklearn_sharpe", sklearn_results["best_sharpe"])
        mlflow.log_param("best_sklearn_model", sklearn_results["best_model"])

        # ── Train Prophet ──────────────────────────────────────────────────
        log.info(f"[{ticker}] Training Prophet...")
        prophet_model, prophet_results = train_prophet(
            df=df, ticker=ticker, n_trials=50
        )

        mlflow.log_metric("prophet_sharpe", prophet_results["best_sharpe"])

        # ── Train LSTM ─────────────────────────────────────────────────────
        log.info(f"[{ticker}] Training LSTM...")
        lstm_model, lstm_results = train_lstm(
            df=df, ticker=ticker, n_trials=30
        )

        mlflow.log_metric("lstm_sharpe", lstm_results["best_sharpe"])

        # ── Generate tuning-set predictions for weight optimisation ────────
        log.info(f"[{ticker}] Generating ensemble predictions for weight tuning...")

        holdout_size  = int(len(df) * wf_config.holdout_pct)
        tuning_data   = df.iloc[:-holdout_size]
        eval_size     = int(len(tuning_data) * 0.3)
        eval_data     = tuning_data.iloc[-eval_size:]

        seq_length    = lstm_results["best_params"]["lstm_seq_length"]
        lstm_threshold= lstm_results["best_params"].get("lstm_threshold", 0.5)

        X_eval        = eval_data[feature_cols].values
        r_eval        = eval_data["log_return"].values

        # Sklearn probs
        sklearn_probs = sklearn_model.predict_proba(X_eval)[:, 1] \
            if hasattr(sklearn_model.named_steps["classifier"], "predict_proba") \
            else sklearn_model.predict(X_eval).astype(float)

        # LSTM probs
        X_seq, _      = build_sequences(X_eval, np.zeros(len(X_eval)), seq_length)
        lstm_probs    = lstm_model.predict(X_seq, verbose=0).flatten()
        r_eval_lstm   = r_eval[seq_length:]
        sklearn_probs_aligned = sklearn_probs[seq_length:]

        # Prophet probs
        prophet_df    = prepare_prophet_df(eval_data)
        horizon       = CONFIG["features"]["prediction_horizon"]
        future_dates  = pd.DataFrame({
            "ds": pd.date_range(
                start   = prophet_df["ds"].iloc[0],
                periods = len(eval_data) + horizon,
                freq    = "B",
            )
        })
        prophet_sigs  = get_prophet_signal(
            prophet_model, future_dates, eval_data, horizon
        )
        prophet_probs = prophet_sigs[seq_length:].astype(float)
        min_len       = min(
            len(prophet_probs), len(lstm_probs), len(sklearn_probs_aligned)
        )

        # ── Tune ensemble weights ──────────────────────────────────────────
        w_prophet, w_lstm, w_sklearn, ensemble_sharpe = tune_ensemble_weights(
            prophet_probs  = prophet_probs[:min_len],
            lstm_probs     = lstm_probs[:min_len],
            sklearn_probs  = sklearn_probs_aligned[:min_len],
            actual_returns = r_eval_lstm[:min_len],
            risk_free_rate = risk_free_rate,
            n_trials       = 200,
        )

        mlflow.log_metric("ensemble_tuning_sharpe", ensemble_sharpe)
        mlflow.log_param("w_prophet", round(w_prophet, 4))
        mlflow.log_param("w_lstm",    round(w_lstm, 4))
        mlflow.log_param("w_sklearn", round(w_sklearn, 4))

        # ── Build and save ensemble ────────────────────────────────────────
        predictor = EnsemblePredictor(
            prophet_model  = prophet_model,
            lstm_model     = lstm_model,
            sklearn_model  = sklearn_model,
            lstm_params    = lstm_results["best_params"],
            feature_cols   = feature_cols,
            weights        = (w_prophet, w_lstm, w_sklearn),
            ticker         = ticker,
        )

        ensemble_path = predictor.save()
        mlflow.log_artifact(str(ensemble_path))

        # ── Evaluate on holdout — ONCE ─────────────────────────────────────
        log.info(f"[{ticker}] Evaluating on final holdout set...")
        holdout = get_holdout(df, wf_config)

        signal, probability = predictor.predict(holdout)

        # Holdout Sharpe
        h_X           = holdout[feature_cols].values
        h_r           = holdout["log_return"].values
        h_preds       = sklearn_model.predict(h_X)
        h_returns     = signal_to_returns(h_preds, h_r)
        holdout_sharpe= compute_sharpe(h_returns, risk_free_rate)

        mlflow.log_metric("holdout_sharpe", holdout_sharpe)

        log.info("=" * 60)
        log.success(
            f"[{ticker}] TRAINING COMPLETE | "
            f"Holdout Sharpe: {holdout_sharpe:.4f} | "
            f"Signal: {signal} ({probability:.3f})"
        )
        log.info("=" * 60)

        return {
            "ticker":          ticker,
            "holdout_sharpe":  holdout_sharpe,
            "ensemble_sharpe": ensemble_sharpe,
            "signal":          signal,
            "probability":     probability,
            "weights": {
                "prophet": w_prophet,
                "lstm":    w_lstm,
                "sklearn": w_sklearn,
            },
            "best_sklearn": sklearn_results["best_model"],
        }


def run_training() -> dict:
    """
    Main entry point — trains all configured tickers.

    Returns:
        Summary of results per ticker
    """

    log.info("=" * 60)
    log.info("STARTING FULL TRAINING PIPELINE")
    log.info("=" * 60)

    tickers = CONFIG["data"]["pairs"]
    results = {}

    for ticker in tickers:
        try:
            results[ticker] = train_ticker(ticker)
        except Exception as e:
            log.error(f"Training failed for {ticker}: {e}")
            results[ticker] = {"status": "failed", "error": str(e)}

    # ── Summary ────────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("TRAINING SUMMARY")
    log.info("=" * 60)

    for ticker, result in results.items():
        if "holdout_sharpe" in result:
            log.success(
                f"✓ {ticker} | "
                f"Holdout Sharpe: {result['holdout_sharpe']:.4f} | "
                f"Signal: {result['signal']} | "
                f"Best Model: {result['best_sklearn']}"
            )
        else:
            log.error(f"✗ {ticker} | {result.get('error', 'Unknown error')}")

    return results


if __name__ == "__main__":
    run_training()
EOF