import numpy as np
import pytest
from src.backtesting.backtest import (
    compute_sharpe_ratio,
    compute_max_drawdown,
    compute_win_rate,
    signals_to_strategy_returns,
    run_backtest,
)

def test_sharpe_ratio_positive_returns():
    returns = np.array([0.01] * 100)
    sharpe = compute_sharpe_ratio(returns)
    assert sharpe > 0

def test_sharpe_ratio_zero_std():
    returns = np.array([0.0] * 100)
    sharpe = compute_sharpe_ratio(returns)
    assert sharpe == 0.0

def test_max_drawdown_is_negative():
    import pandas as pd
    cumulative = pd.Series([1.0, 1.1, 0.9, 0.95, 1.2])
    dd = compute_max_drawdown(cumulative)
    assert dd < 0

def test_win_rate_all_positive():
    returns = np.array([0.01] * 100)
    wr = compute_win_rate(returns)
    assert wr == 1.0

def test_win_rate_all_negative():
    returns = np.array([-0.01] * 100)
    wr = compute_win_rate(returns)
    assert wr == 0.0

def test_strategy_returns_length():
    preds = np.array([1, 0, 1, 1, 0])
    actual = np.array([0.01, -0.01, 0.02, -0.005, 0.01])
    result = signals_to_strategy_returns(preds, actual)
    assert len(result) == len(preds)

def test_run_backtest_returns_results():
    np.random.seed(42)
    preds = np.random.randint(0, 2, 200)
    returns = np.random.normal(0.0003, 0.01, 200)
    results = run_backtest(preds, returns, ticker='TEST')
    assert results.ticker == 'TEST'
    assert isinstance(results.sharpe_ratio, float)
    assert isinstance(results.max_drawdown, float)
    assert isinstance(results.win_rate, float)

def test_run_backtest_insufficient_data():
    from src.utils.exceptions import BacktestError
    preds = np.array([1, 0, 1])
    returns = np.array([0.01, -0.01, 0.02])
    with pytest.raises(BacktestError):
        run_backtest(preds, returns, ticker='TEST')
