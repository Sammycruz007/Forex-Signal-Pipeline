import pandas as pd
import numpy as np
import pandas_ta_classic as ta
from pathlib import Path
from typing import Tuple

from src.utils.logger import log
from src.utils.exceptions import (
    FeatureEngineeringError,
    DataValidationError,
)
from src.utils.config import CONFIG


"""
Feature Engineering Module

Reads validated raw OHLCV data and computes technical indicators.
Output is saved to data/features/ as Parquet files ready for model training.

Design rules:
- NEVER use future data to compute a feature (lookahead bias = invalid backtest)
- Every indicator is computed on close/high/low/volume only
- The target variable (5-day forward return) is computed here
- NaN rows from indicator warmup periods are dropped cleanly
"""


# ── Feature Engineering Functions ─────────────────────────────────────────────

def add_momentum_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Add momentum-based technical indicators.
    These measure the speed and strength of price movement.

    Indicators:
    - RSI: Relative Strength Index — overbought/oversold signal
    - MACD: Moving Average Convergence Divergence — trend momentum
    - ROC: Rate of Change — raw momentum over N periods
    """

    log.debug("Computing momentum features...")

    # RSI — values above 70 = overbought, below 30 = oversold
    df["rsi"] = ta.rsi(df["Close"], length=cfg["rsi_period"])

    # MACD — when MACD crosses signal line, trend may be changing
    macd = ta.macd(
        df["Close"],
        fast=cfg["macd_fast"],
        slow=cfg["macd_slow"],
        signal=cfg["macd_signal"],
    )
    df["macd"]        = macd[f"MACD_{cfg['macd_fast']}_{cfg['macd_slow']}_{cfg['macd_signal']}"]
    df["macd_signal"] = macd[f"MACDs_{cfg['macd_fast']}_{cfg['macd_slow']}_{cfg['macd_signal']}"]
    df["macd_hist"]   = macd[f"MACDh_{cfg['macd_fast']}_{cfg['macd_slow']}_{cfg['macd_signal']}"]

    # ROC — percentage change over 10 periods
    df["roc_10"] = ta.roc(df["Close"], length=10)

    return df


def add_trend_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Add trend-following indicators.
    These identify the direction and strength of the prevailing trend.

    Indicators:
    - SMA: Simple Moving Average (10, 20, 50 day)
    - EMA: Exponential Moving Average (12, 26 day)
    - Price/SMA ratio: How far price has deviated from its average
    - ADX: Average Directional Index — trend strength (not direction)
    """

    log.debug("Computing trend features...")

    # Simple Moving Averages
    for period in cfg["sma_periods"]:
        df[f"sma_{period}"] = ta.sma(df["Close"], length=period)
        # Price ratio: above 1.0 means price is above its average
        df[f"price_sma_{period}_ratio"] = df["Close"] / df[f"sma_{period}"]

    # Exponential Moving Averages — react faster than SMA
    df["ema_12"] = ta.ema(df["Close"], length=12)
    df["ema_26"] = ta.ema(df["Close"], length=26)

    # EMA crossover signal — positive when short EMA > long EMA (bullish)
    df["ema_crossover"] = df["ema_12"] - df["ema_26"]

    # ADX — above 25 means strong trend, below 25 means ranging market
    adx = ta.adx(df["High"], df["Low"], df["Close"], length=14)
    df["adx"] = adx["ADX_14"]

    return df


def add_volatility_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Add volatility indicators.
    These measure how much prices are moving — critical for risk management.

    Indicators:
    - Bollinger Bands: Price envelope based on standard deviation
    - ATR: Average True Range — absolute volatility measure
    - BB Width: How wide the bands are — measures volatility regime
    - BB Position: Where price sits within the bands (0=lower, 1=upper)
    """

    log.debug("Computing volatility features...")

    # Bollinger Bands
    bb = ta.bbands(df["Close"], length=cfg["bb_period"])
    df["bb_upper"] = bb[f"BBU_{cfg['bb_period']}_2.0"]
    df["bb_mid"]   = bb[f"BBM_{cfg['bb_period']}_2.0"]
    df["bb_lower"] = bb[f"BBL_{cfg['bb_period']}_2.0"]

    # BB Width — wide bands = high volatility, narrow = low volatility
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]

    # BB Position — where is price within the bands (0.0 to 1.0)
    bb_range = df["bb_upper"] - df["bb_lower"]
    df["bb_position"] = (df["Close"] - df["bb_lower"]) / bb_range.replace(0, np.nan)

    # ATR — average pip movement per day, used for position sizing
    df["atr"] = ta.atr(
        df["High"], df["Low"], df["Close"],
        length=cfg["atr_period"]
    )

    # Normalised ATR — ATR as % of price (comparable across pairs)
    df["atr_pct"] = df["atr"] / df["Close"]

    return df


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add price-derived features.
    These capture raw price behaviour without indicators.

    Features:
    - Daily return: Percentage change from previous close
    - Log return: Log of price ratio (better statistical properties)
    - High-Low range: Intraday volatility
    - Gap: Overnight price gap from previous close
    """

    log.debug("Computing price-derived features...")

    # Daily percentage return
    df["daily_return"] = df["Close"].pct_change()

    # Log return — preferred for financial modelling (additive over time)
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))

    # Intraday range as % of open
    df["hl_range_pct"] = (df["High"] - df["Low"]) / df["Open"]

    # Overnight gap — did price jump from yesterday's close?
    df["gap_pct"] = (df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1)

    return df


def add_target_variable(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    Add the prediction target variable.

    Target: 5-day forward return direction
    - 1  → Price will be HIGHER in 5 days (BUY signal)
    - 0  → Price will be LOWER or flat in 5 days (SELL/HOLD signal)

    IMPORTANT: We shift BACKWARDS to get future returns.
    This is correct — we are labelling today with what happens next.
    There is NO lookahead bias here because during inference,
    we never have this column — it gets predicted, not computed.

    Args:
        df:      Feature DataFrame
        horizon: Number of days ahead to predict (from config)
    """

    log.debug(f"Computing {horizon}-day forward return target...")

    # Forward return: what will close be in `horizon` days vs today
    df["forward_return"] = df["Close"].shift(-horizon) / df["Close"] - 1

    # Binary classification target: 1 = up, 0 = down/flat
    df["target"] = (df["forward_return"] > 0).astype(int)

    # Drop the last `horizon` rows — they have no valid target
    # (we don't know the future yet for the most recent rows)
    df = df.iloc[:-horizon]

    log.debug(
        f"Target distribution: "
        f"UP={df['target'].sum()} | "
        f"DOWN={(df['target']==0).sum()}"
    )

    return df


def build_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Master function — applies all feature engineering steps in sequence.

    Args:
        df:     Validated raw OHLCV DataFrame
        ticker: Ticker name for logging

    Returns:
        Feature-rich DataFrame ready for model training

    Raises:
        FeatureEngineeringError: If any step fails or output is invalid
    """

    log.info(f"Building features for {ticker} | Input shape: {df.shape}")

    cfg = CONFIG["features"]
    horizon = cfg["prediction_horizon"]

    try:
        # Apply feature groups in sequence
        df = add_momentum_features(df, cfg)
        df = add_trend_features(df, cfg)
        df = add_volatility_features(df, cfg)
        df = add_price_features(df)
        df = add_target_variable(df, horizon)

    except Exception as e:
        raise FeatureEngineeringError(
            f"Feature engineering failed for {ticker}",
            details={"error": str(e), "ticker": ticker}
        )

    # ── Drop NaN rows from indicator warmup periods ────────────────────────
    rows_before = len(df)
    df = df.dropna()
    dropped = rows_before - len(df)

    if dropped > 0:
        log.warning(
            f"Dropped {dropped} NaN rows from warmup periods in {ticker}"
        )

    # ── Validate output ────────────────────────────────────────────────────
    if len(df) < 100:
        raise DataValidationError(
            f"Too few rows after feature engineering for {ticker}",
            details={"rows_remaining": len(df)}
        )

    if "target" not in df.columns:
        raise FeatureEngineeringError(
            "Target column missing after feature engineering",
            details={"ticker": ticker}
        )

    log.success(
        f"Features built for {ticker} | "
        f"Output shape: {df.shape} | "
        f"Features: {len(df.columns)} columns"
    )

    return df


def save_features(df: pd.DataFrame, ticker: str, features_path: str) -> Path:
    """
    Save engineered features to disk as a Parquet file.

    Args:
        df:            Feature DataFrame
        ticker:        Ticker symbol
        features_path: Output directory from config

    Returns:
        Path to saved file
    """

    clean_ticker = ticker.replace("=X", "").replace("/", "").strip()
    output_dir = Path(features_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    file_path = output_dir / f"{clean_ticker}_features.parquet"

    try:
        df.to_parquet(file_path, index=True)
        log.success(f"Features saved → {file_path}")
    except Exception as e:
        raise FeatureEngineeringError(
            f"Failed to save features for {ticker}",
            details={"path": str(file_path), "error": str(e)}
        )

    return file_path


def run_feature_engineering() -> dict:
    """
    Main entry point for feature engineering pipeline.
    Reads all raw data files and builds features for each ticker.

    Returns:
        Summary dict with results per ticker
    """

    log.info("=" * 60)
    log.info("STARTING FEATURE ENGINEERING PIPELINE")
    log.info("=" * 60)

    tickers       = CONFIG["data"]["pairs"]
    raw_path      = CONFIG["data"]["raw_path"]
    features_path = CONFIG["data"]["features_path"]

    results = {}
    failed  = []

    for ticker in tickers:
        clean_ticker = ticker.replace("=X", "").replace("/", "").strip()
        raw_file = Path(raw_path) / f"{clean_ticker}_raw.parquet"

        log.info(f"Processing {ticker}...")

        # ── Check raw file exists ──────────────────────────────────────────
        if not raw_file.exists():
            log.error(f"Raw file not found for {ticker}: {raw_file}")
            results[ticker] = {
                "status": "failed",
                "error": f"Raw file not found: {raw_file}"
            }
            failed.append(ticker)
            continue

        try:
            # Load raw data
            raw_df = pd.read_parquet(raw_file)
            log.info(f"Loaded raw data for {ticker} | Shape: {raw_df.shape}")

            # Build features
            feature_df = build_features(df=raw_df.copy(), ticker=ticker)

            # Save features
            saved_path = save_features(
                df=feature_df,
                ticker=ticker,
                features_path=features_path,
            )

            results[ticker] = {
                "status":   "success",
                "rows":     len(feature_df),
                "features": len(feature_df.columns),
                "path":     str(saved_path),
            }

        except Exception as e:
            log.error(f"Feature engineering failed for {ticker}: {e}")
            results[ticker] = {"status": "failed", "error": str(e)}
            failed.append(ticker)

    # ── Summary ────────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("FEATURE ENGINEERING SUMMARY")
    log.info("=" * 60)

    for ticker, result in results.items():
        if result["status"] == "success":
            log.success(
                f"✓ {ticker} | "
                f"{result['rows']} rows | "
                f"{result['features']} features"
            )
        else:
            log.error(f"✗ {ticker} | {result['error']}")

    if len(failed) == len(tickers):
        raise FeatureEngineeringError(
            "Feature engineering failed for all tickers.",
            details={"failed": failed}
        )

    log.info("Feature engineering complete.")
    return results


# ── Script Entry Point ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_feature_engineering()