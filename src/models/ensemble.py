import numpy as np
import pandas as pd
import optuna
import joblib
from pathlib import Path
from typing import Tuple

from src.models.sklearn_model import compute_sharpe, signal_to_returns
from src.utils.logger import log
from src.utils.exceptions import ModelTrainingError
from src.utils.config import CONFIG

optuna.logging.set_verbosity(optuna.logging.WARNING)


"""
Ensemble Model — Combines Prophet + LSTM + Best Sklearn

After all three models are individually tuned, this module:
1. Generates out-of-fold predictions from each model
2. Uses Optuna to find optimal ensemble weights (w1, w2, w3)
3. Produces the final BUY / SELL / HOLD signal

Weighting strategy:
- Weights are constrained to sum to 1.0
- Optuna maximises Sharpe of the weighted combination
- This is a learned ensemble — not a naive average
"""


# ── Signal Label ───────────────────────────────────────────────────────────────

def probability_to_signal(probability: float, threshold: float = 0.5) -> str:
    """
    Convert ensemble probability to a human-readable trading signal.

    Args:
        probability: Ensemble output (0.0 to 1.0)
        threshold:   Decision boundary

    Returns:
        "BUY", "SELL", or "HOLD"
    """
    if probability > threshold + 0.1:
        return "BUY"
    elif probability < threshold - 0.1:
        return "SELL"
    else:
        return "HOLD"


# ── Weight Tuner ───────────────────────────────────────────────────────────────

def tune_ensemble_weights(
    prophet_probs: np.ndarray,
    lstm_probs: np.ndarray,
    sklearn_probs: np.ndarray,
    actual_returns: np.ndarray,
    risk_free_rate: float,
    n_trials: int = 200,
) -> Tuple[float, float, float, float]:
    """
    Use Optuna to find optimal weights for the ensemble.

    Weights are constrained: w1 + w2 + w3 = 1.0
    We use a Dirichlet-like parameterisation to enforce this.

    Args:
        prophet_probs:  Prophet predicted probabilities
        lstm_probs:     LSTM predicted probabilities
        sklearn_probs:  Sklearn predicted probabilities
        actual_returns: True strategy returns for evaluation
        risk_free_rate: Annual risk-free rate
        n_trials:       Optuna trials for weight search

    Returns:
        Tuple of (w_prophet, w_lstm, w_sklearn, best_sharpe)
    """

    def weight_objective(trial: optuna.Trial) -> float:
        w1 = trial.suggest_float("w_prophet", 0.05, 0.70)
        w2 = trial.suggest_float("w_lstm",    0.05, 0.70)
        w3 = 1.0 - w1 - w2

        # Penalise if w3 is out of valid range
        if w3 < 0.05 or w3 > 0.70:
            return -999.0

        # Weighted ensemble probability
        ensemble_probs = w1 * prophet_probs + w2 * lstm_probs + w3 * sklearn_probs

        # Convert to binary signals
        preds            = (ensemble_probs >= 0.5).astype(int)
        strategy_returns = signal_to_returns(preds, actual_returns)
        return compute_sharpe(strategy_returns, risk_free_rate)

    study = optuna.create_study(
        direction = "maximize",
        sampler   = optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(weight_objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_trial.params
    w1   = best["w_prophet"]
    w2   = best["w_lstm"]
    w3   = 1.0 - w1 - w2

    log.success(
        f"Optimal weights | "
        f"Prophet: {w1:.3f} | "
        f"LSTM: {w2:.3f} | "
        f"Sklearn: {w3:.3f} | "
        f"Sharpe: {study.best_value:.4f}"
    )

    return w1, w2, w3, study.best_value


# ── Ensemble Predictor ─────────────────────────────────────────────────────────

class EnsemblePredictor:
    """
    Production ensemble predictor.
    Holds all three models and their learned weights.
    Called by FastAPI for inference.

    Usage:
        predictor = EnsemblePredictor.load(ticker)
        signal, probability = predictor.predict(feature_df)
    """

    def __init__(
        self,
        prophet_model,
        lstm_model,
        sklearn_model,
        lstm_params: dict,
        feature_cols: list,
        weights: Tuple[float, float, float],
        ticker: str,
    ):
        self.prophet_model  = prophet_model
        self.lstm_model     = lstm_model
        self.sklearn_model  = sklearn_model
        self.lstm_params    = lstm_params
        self.feature_cols   = feature_cols
        self.w_prophet      = weights[0]
        self.w_lstm         = weights[1]
        self.w_sklearn      = weights[2]
        self.ticker         = ticker

    def predict(self, df: pd.DataFrame) -> Tuple[str, float]:
        """
        Generate ensemble prediction for the most recent data.

        Args:
            df: Feature DataFrame (must have enough rows for LSTM sequence)

        Returns:
            Tuple of (signal_string, probability)
            signal_string: "BUY", "SELL", or "HOLD"
            probability:   Ensemble confidence (0.0 to 1.0)
        """
        from src.models.lstm_model import build_sequences
        from src.models.prophet_model import prepare_prophet_df, get_prophet_signal

        horizon    = CONFIG["features"]["prediction_horizon"]
        seq_length = self.lstm_params["lstm_seq_length"]
        threshold  = self.lstm_params.get("lstm_threshold", 0.5)

        # ── Sklearn prediction ─────────────────────────────────────────────
        X_latest       = df[self.feature_cols].values[-1].reshape(1, -1)
        sklearn_prob   = float(
            self.sklearn_model.predict_proba(X_latest)[0][1]
            if hasattr(self.sklearn_model.named_steps["classifier"], "predict_proba")
            else float(self.sklearn_model.predict(X_latest)[0])
        )

        # ── LSTM prediction ────────────────────────────────────────────────
        X_seq = df[self.feature_cols].values[-seq_length:].reshape(
            1, seq_length, len(self.feature_cols)
        )
        lstm_prob = float(self.lstm_model.predict(X_seq, verbose=0)[0][0])

        # ── Prophet prediction ─────────────────────────────────────────────
        prophet_df   = prepare_prophet_df(df)
        future_dates = pd.DataFrame({
            "ds": pd.date_range(
                start   = prophet_df["ds"].iloc[-1],
                periods = horizon + 1,
                freq    = "B",
            )
        })
        forecast     = self.prophet_model.predict(future_dates)
        current_px   = df["Close"].iloc[-1]
        future_px    = forecast["yhat"].iloc[-1]
        prophet_prob = 1.0 if future_px > current_px else 0.0

        # ── Weighted ensemble ──────────────────────────────────────────────
        ensemble_prob = (
            self.w_prophet * prophet_prob
            + self.w_lstm   * lstm_prob
            + self.w_sklearn * sklearn_prob
        )

        signal = probability_to_signal(ensemble_prob)

        log.info(
            f"Ensemble prediction | {self.ticker} | "
            f"Prophet: {prophet_prob:.3f} | "
            f"LSTM: {lstm_prob:.3f} | "
            f"Sklearn: {sklearn_prob:.3f} | "
            f"Ensemble: {ensemble_prob:.3f} | "
            f"Signal: {signal}"
        )

        return signal, ensemble_prob

    def save(self, output_dir: str = "models/ensemble") -> Path:
        """Save the entire ensemble predictor to disk."""
        clean_ticker = self.ticker.replace("=X", "").replace("/", "")
        save_dir     = Path(output_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / f"{clean_ticker}_ensemble.pkl"
        joblib.dump(self, path)
        log.success(f"Ensemble predictor saved → {path}")
        return path

    @classmethod
    def load(cls, ticker: str, model_dir: str = "models/ensemble") -> "EnsemblePredictor":
        """Load a saved ensemble predictor from disk."""
        clean_ticker = ticker.replace("=X", "").replace("/", "")
        path = Path(model_dir) / f"{clean_ticker}_ensemble.pkl"

        if not path.exists():
            raise FileNotFoundError(f"Ensemble model not found: {path}")

        predictor = joblib.load(path)
        log.info(f"Ensemble predictor loaded from {path}")
        return predictor
EOF