import numpy as np
import pandas as pd
import optuna
import joblib
from pathlib import Path
from typing import List, Tuple

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

from src.models.walk_forward import (
    WalkForwardFold,
    WalkForwardConfig,
    generate_folds,
    get_feature_columns,
)
from src.models.sklearn_model import compute_sharpe, signal_to_returns
from src.utils.logger import log
from src.utils.exceptions import ModelTrainingError
from src.utils.config import CONFIG

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Suppress TensorFlow logs — only show errors
tf.get_logger().setLevel("ERROR")

"""
LSTM Model with Optuna Tuning

Long Short-Term Memory networks capture sequential dependencies
in time series that tree models and Prophet cannot.

Key design decisions:
- Sequence-based input: model sees last N days to predict next direction
- Early stopping per fold: prevents overfitting within each fold
- Optuna tunes architecture AND training hyperparameters
- Model outputs probability (0-1) — converted to binary signal at threshold
"""


# ── Sequence Builder ───────────────────────────────────────────────────────────

def build_sequences(
    X: np.ndarray,
    y: np.ndarray,
    seq_length: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transform flat feature matrix into sequences for LSTM input.

    Standard LSTM input shape: (samples, timesteps, features)

    Example with seq_length=5:
    Row 5's input = [Row0, Row1, Row2, Row3, Row4] → predict Row5's target

    IMPORTANT: This preserves temporal order — no shuffling.

    Args:
        X:          Feature matrix (n_samples, n_features)
        y:          Target array (n_samples,)
        seq_length: Number of timesteps per sequence

    Returns:
        Tuple of (X_seq, y_seq) with shapes:
        X_seq: (n_samples - seq_length, seq_length, n_features)
        y_seq: (n_samples - seq_length,)
    """
    X_seq, y_seq = [], []

    for i in range(seq_length, len(X)):
        X_seq.append(X[i - seq_length:i])
        y_seq.append(y[i])

    return np.array(X_seq), np.array(y_seq)


# ── Model Builder ──────────────────────────────────────────────────────────────

def build_lstm(
    trial: optuna.Trial,
    seq_length: int,
    n_features: int,
) -> Sequential:
    """
    Build LSTM architecture based on Optuna trial suggestions.

    Tuned parameters:
    - n_layers:      1 or 2 LSTM layers (deeper is not always better)
    - units_l1/l2:   Neurons per LSTM layer
    - dropout:       Regularisation (prevents memorising training data)
    - learning_rate: Step size for gradient descent

    Args:
        trial:      Optuna trial
        seq_length: Input sequence length
        n_features: Number of input features

    Returns:
        Compiled Keras Sequential model
    """

    n_layers = trial.suggest_int("lstm_n_layers", 1, 2)
    dropout  = trial.suggest_float("lstm_dropout", 0.1, 0.5)
    lr       = trial.suggest_float("lstm_lr", 1e-4, 1e-2, log=True)
    units_l1 = trial.suggest_int("lstm_units_l1", 32, 128)

    model = Sequential()
    model.add(Input(shape=(seq_length, n_features)))

    if n_layers == 1:
        model.add(LSTM(units_l1, return_sequences=False))
        model.add(Dropout(dropout))
    else:
        units_l2 = trial.suggest_int("lstm_units_l2", 16, 64)
        model.add(LSTM(units_l1, return_sequences=True))
        model.add(Dropout(dropout))
        model.add(LSTM(units_l2, return_sequences=False))
        model.add(Dropout(dropout))

    # Output layer: sigmoid → probability of price going UP
    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        optimizer = Adam(learning_rate=lr),
        loss      = "binary_crossentropy",
        metrics   = ["accuracy"],
    )

    return model


# ── Objective Function ─────────────────────────────────────────────────────────

def lstm_objective(
    trial: optuna.Trial,
    folds: List[WalkForwardFold],
    feature_cols: List[str],
    risk_free_rate: float,
) -> float:
    """
    Optuna objective for LSTM tuning.
    Evaluates mean Sharpe across all walk-forward folds.

    Args:
        trial:          Optuna trial
        folds:          Walk-forward folds
        feature_cols:   Feature column names
        risk_free_rate: Annual risk-free rate

    Returns:
        Mean Sharpe Ratio across folds
    """

    seq_length = trial.suggest_int("lstm_seq_length", 20, 60)
    batch_size = trial.suggest_categorical("lstm_batch_size", [16, 32, 64])
    epochs     = trial.suggest_int("lstm_epochs", 20, 60)
    threshold  = trial.suggest_float("lstm_threshold", 0.4, 0.6)

    sharpes = []

    for fold in folds:
        try:
            X_train_raw = fold.train[feature_cols].values
            y_train_raw = fold.train["target"].values

            eval_data   = pd.concat([fold.val, fold.test])
            X_eval_raw  = eval_data[feature_cols].values
            y_eval_raw  = eval_data["target"].values
            r_eval      = eval_data["log_return"].values

            # Build sequences
            X_train, y_train = build_sequences(X_train_raw, y_train_raw, seq_length)
            X_eval,  y_eval  = build_sequences(X_eval_raw,  y_eval_raw,  seq_length)

            if len(X_train) < 10 or len(X_eval) < 5:
                sharpes.append(-1.0)
                continue

            # Build and train model
            model = build_lstm(trial, seq_length, len(feature_cols))

            early_stop = EarlyStopping(
                monitor   = "val_loss",
                patience  = 5,
                restore_best_weights = True,
            )

            model.fit(
                X_train, y_train,
                epochs          = epochs,
                batch_size      = batch_size,
                validation_split= 0.1,
                callbacks       = [early_stop],
                verbose         = 0,
            )

            # Predict and convert to binary signals
            probs   = model.predict(X_eval, verbose=0).flatten()
            preds   = (probs >= threshold).astype(int)

            # Align returns with sequence offset
            r_eval_aligned   = r_eval[seq_length:]
            strategy_returns = signal_to_returns(preds, r_eval_aligned)
            sharpe           = compute_sharpe(strategy_returns, risk_free_rate)
            sharpes.append(sharpe)

            # Free GPU memory between folds
            del model
            tf.keras.backend.clear_session()

        except Exception as e:
            log.warning(f"LSTM trial {trial.number} fold {fold.fold_id} failed: {e}")
            sharpes.append(-1.0)

    mean_sharpe = float(np.mean(sharpes))
    log.debug(f"LSTM Trial {trial.number:>4} | Mean Sharpe: {mean_sharpe:.4f}")
    return mean_sharpe


# ── Main Training Function ─────────────────────────────────────────────────────

def train_lstm(
    df: pd.DataFrame,
    ticker: str,
    n_trials: int = 30,
    output_dir: str = "models/lstm",
) -> Tuple[Sequential, dict]:
    """
    Tune and train LSTM using Optuna + walk-forward validation.

    Note: n_trials is lower than sklearn (30 vs 100) because
    each LSTM trial trains multiple Keras models — it is expensive.

    Args:
        df:         Feature DataFrame
        ticker:     Ticker symbol
        n_trials:   Number of Optuna trials
        output_dir: Where to save model weights

    Returns:
        Tuple of (trained_lstm_model, results_dict)

    Raises:
        ModelTrainingError: If training fails
    """

    log.info("=" * 60)
    log.info(f"LSTM MODEL TRAINING — {ticker}")
    log.info(f"Trials: {n_trials}")
    log.info("=" * 60)

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
            f"Failed to generate folds for LSTM — {ticker}",
            details={"error": str(e)}
        )

    # ── Optuna study ───────────────────────────────────────────────────────
    study = optuna.create_study(
        direction  = "maximize",
        study_name = f"lstm_{ticker}",
        sampler    = optuna.samplers.TPESampler(seed=42),
    )

    study.optimize(
        lambda trial: lstm_objective(
            trial, folds, feature_cols, risk_free_rate
        ),
        n_trials          = n_trials,
        show_progress_bar = True,
    )

    best_params = study.best_trial.params
    best_sharpe = study.best_trial.value

    log.success(f"Best LSTM Sharpe: {best_sharpe:.4f}")
    log.info(f"Best params: {best_params}")

    # ── Retrain best architecture on full tuning data ──────────────────────
    holdout_size = int(len(df) * wf_config.holdout_pct)
    tuning_data  = df.iloc[:-holdout_size]

    seq_length   = best_params["lstm_seq_length"]
    batch_size   = best_params["lstm_batch_size"]
    epochs       = best_params["lstm_epochs"]

    X_full = tuning_data[feature_cols].values
    y_full = tuning_data["target"].values
    X_seq, y_seq = build_sequences(X_full, y_full, seq_length)

    final_model = build_lstm(
        study.best_trial, seq_length, len(feature_cols)
    )

    early_stop = EarlyStopping(
        monitor              = "val_loss",
        patience             = 5,
        restore_best_weights = True,
    )

    final_model.fit(
        X_seq, y_seq,
        epochs           = epochs,
        batch_size       = batch_size,
        validation_split = 0.1,
        callbacks        = [early_stop],
        verbose          = 1,
    )

    # ── Save model ─────────────────────────────────────────────────────────
    clean_ticker = ticker.replace("=X", "").replace("/", "")
    save_dir     = Path(output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    model_path   = save_dir / f"{clean_ticker}_lstm"

    final_model.save(str(model_path))
    log.success(f"LSTM model saved → {model_path}")

    # Save best params separately for inference
    params_path = save_dir / f"{clean_ticker}_lstm_params.pkl"
    joblib.dump(best_params, params_path)

    results = {
        "ticker":       ticker,
        "best_sharpe":  best_sharpe,
        "best_params":  best_params,
        "n_trials":     n_trials,
        "n_folds":      len(folds),
        "model_path":   str(model_path),
        "feature_cols": feature_cols,
    }

    return final_model, results
EOF