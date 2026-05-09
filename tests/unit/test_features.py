import pandas as pd
import numpy as np
import pytest
from src.features.engineer import (
    add_momentum_features,
    add_trend_features,
    add_volatility_features,
    add_price_features,
    add_target_variable,
)

def make_sample_df(n=200):
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=n, freq='B')
    close = 1.1 + np.cumsum(np.random.normal(0, 0.001, n))
    df = pd.DataFrame({
        'Open':   close * 0.999,
        'High':   close * 1.002,
        'Low':    close * 0.998,
        'Close':  close,
        'Volume': np.random.randint(1000, 5000, n).astype(float),
    }, index=dates)
    return df

def test_momentum_features_added():
    from src.utils.config import CONFIG
    df = make_sample_df()
    cfg = CONFIG['features']
    result = add_momentum_features(df.copy(), cfg)
    assert 'rsi' in result.columns
    assert 'macd' in result.columns
    assert 'macd_signal' in result.columns

def test_trend_features_added():
    from src.utils.config import CONFIG
    df = make_sample_df()
    cfg = CONFIG['features']
    result = add_trend_features(df.copy(), cfg)
    assert 'sma_10' in result.columns
    assert 'ema_12' in result.columns
    assert 'adx' in result.columns

def test_volatility_features_added():
    from src.utils.config import CONFIG
    df = make_sample_df()
    cfg = CONFIG['features']
    result = add_volatility_features(df.copy(), cfg)
    assert 'atr' in result.columns
    assert 'bb_width' in result.columns
    assert 'bb_position' in result.columns

def test_target_variable_binary():
    df = make_sample_df()
    df = add_price_features(df)
    df = add_target_variable(df, horizon=5)
    assert 'target' in df.columns
    assert set(df['target'].unique()).issubset({0, 1})

def test_no_lookahead_bias():
    df = make_sample_df(n=100)
    df = add_price_features(df)
    df = add_target_variable(df, horizon=5)
    assert df['target'].iloc[-1] in [0, 1]
    assert len(df) == 95
