import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple
import json
from src.utils.logger import log
from src.utils.exceptions import BacktestError, SharpeThresholdError
from src.utils.config import CONFIG

@dataclass
class BacktestResults:
    ticker: str
    sharpe_ratio: float
    max_drawdown: float
    total_return: float
    win_rate: float
    n_trades: int
    avg_return_per_trade: float
    passed_gate: bool

    def __str__(self):
        status = 'PASSED' if self.passed_gate else 'FAILED'
        return (
            f'Backtest Results [{self.ticker}] | {status}\n'
            f'  Sharpe Ratio:         {self.sharpe_ratio:.4f}\n'
            f'  Max Drawdown:         {self.max_drawdown:.2%}\n'
            f'  Total Return:         {self.total_return:.2%}\n'
            f'  Win Rate:             {self.win_rate:.2%}\n'
            f'  Number of Trades:     {self.n_trades}\n'
            f'  Avg Return per Trade: {self.avg_return_per_trade:.4%}\n'
        )

    def to_dict(self):
        return {
            'ticker': self.ticker,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'total_return': self.total_return,
            'win_rate': self.win_rate,
            'n_trades': self.n_trades,
            'avg_return_per_trade': self.avg_return_per_trade,
            'passed_gate': self.passed_gate,
        }

def compute_sharpe_ratio(returns, risk_free_rate=0.0):
    if len(returns) < 2:
        return 0.0
    excess = returns - risk_free_rate / 252
    std = np.std(excess)
    if std == 0:
        return 0.0
    return float(np.mean(excess) / std * np.sqrt(252))

def compute_max_drawdown(cumulative_returns):
    if len(cumulative_returns) < 2:
        return 0.0
    peak = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - peak) / peak
    return float(drawdown.min())

def compute_win_rate(strategy_returns):
    if len(strategy_returns) == 0:
        return 0.0
    wins = (strategy_returns > 0).sum()
    return float(wins / len(strategy_returns))

def signals_to_strategy_returns(predictions, actual_returns, transaction_cost=0.0002):
    positions = np.where(predictions == 1, 1, -1)
    position_changes = np.diff(positions, prepend=0)
    costs = np.abs(position_changes) * transaction_cost
    strategy_returns = positions * actual_returns - costs
    return strategy_returns

def run_backtest(
    predictions,
    actual_returns,
    ticker,
    risk_free_rate=None,
    sharpe_threshold=None,
    transaction_cost=0.0002,
):
    if risk_free_rate is None:
        risk_free_rate = CONFIG['backtesting']['risk_free_rate']
    if sharpe_threshold is None:
        sharpe_threshold = CONFIG['backtesting']['sharpe_threshold']
    log.info(f'Running backtest for {ticker} | Signals: {len(predictions)}')
    if len(predictions) != len(actual_returns):
        raise BacktestError(
            'Predictions and returns length mismatch',
            details={
                'predictions_len': len(predictions),
                'returns_len': len(actual_returns)
            }
        )
    if len(predictions) < 30:
        raise BacktestError(
            'Insufficient data for meaningful backtest',
            details={'n_samples': len(predictions), 'minimum': 30}
        )
    strategy_returns = signals_to_strategy_returns(
        predictions, actual_returns, transaction_cost
    )
    cumulative = pd.Series((1 + strategy_returns).cumprod())
    sharpe = compute_sharpe_ratio(strategy_returns, risk_free_rate)
    max_dd = compute_max_drawdown(cumulative)
    total_return = float(cumulative.iloc[-1] - 1)
    win_rate = compute_win_rate(strategy_returns)
    n_trades = int(np.sum(np.abs(np.diff(np.where(predictions == 1, 1, -1), prepend=0)) > 0))
    avg_return = float(np.mean(strategy_returns))
    passed = sharpe >= sharpe_threshold
    results = BacktestResults(
        ticker=ticker,
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        total_return=total_return,
        win_rate=win_rate,
        n_trades=n_trades,
        avg_return_per_trade=avg_return,
        passed_gate=passed,
    )
    log.info(str(results))
    if passed:
        log.success(
            f'Backtest PASSED | '
            f'Sharpe {sharpe:.4f} >= threshold {sharpe_threshold}'
        )
    else:
        log.error(
            f'Backtest FAILED | '
            f'Sharpe {sharpe:.4f} < threshold {sharpe_threshold}'
        )
    return results

def save_backtest_results(results, output_dir='logs'):
    Path(output_dir).mkdir(exist_ok=True)
    output_path = Path(output_dir) / f'backtest_{results.ticker}.json'
    with open(output_path, 'w') as f:
        json.dump(results.to_dict(), f, indent=2)
    log.success(f'Backtest results saved to {output_path}')
    return output_path

def run_sharpe_gate(results):
    sharpe_threshold = CONFIG['backtesting']['sharpe_threshold']
    if not results.passed_gate:
        raise SharpeThresholdError(
            f'Model failed Sharpe gate for {results.ticker}',
            details={
                'achieved_sharpe': results.sharpe_ratio,
                'required_sharpe': sharpe_threshold,
                'shortfall': sharpe_threshold - results.sharpe_ratio,
            }
        )
    log.success(
        f'Sharpe gate PASSED for {results.ticker} | '
        f'Sharpe: {results.sharpe_ratio:.4f}'
    )
    return True
