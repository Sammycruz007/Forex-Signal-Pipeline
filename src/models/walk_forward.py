import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List
from src.utils.logger import log
from src.utils.exceptions import BacktestError
from src.utils.config import CONFIG

@dataclass
class WalkForwardFold:
    fold_id: int
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_end: pd.Timestamp

    def __repr__(self):
        return (
            f'Fold {self.fold_id} | '
            f'Train: {self.train_start.date()} to {self.train_end.date()} '
            f'({len(self.train)} rows) | '
            f'Val: {len(self.val)} rows | '
            f'Test: {len(self.test)} rows'
        )

@dataclass
class WalkForwardConfig:
    train_window: int = 500
    val_window: int = 63
    test_window: int = 21
    step_size: int = 21
    holdout_pct: float = 0.20

def generate_folds(df, config):
    holdout_size = int(len(df) * config.holdout_pct)
    tuning_zone = df.iloc[:-holdout_size]
    holdout_zone = df.iloc[-holdout_size:]
    log.info(
        f'Data split | '
        f'Tuning zone: {len(tuning_zone)} rows | '
        f'Holdout: {len(holdout_zone)} rows '
        f'({tuning_zone.index[-1].date()} to {holdout_zone.index[-1].date()})'
    )
    min_required = config.train_window + config.val_window + config.test_window
    if len(tuning_zone) < min_required:
        raise BacktestError(
            'Insufficient data for walk-forward validation',
            details={
                'rows_available': len(tuning_zone),
                'rows_required': min_required,
                'suggestion': 'Reduce train_window or increase period_years in config'
            }
        )
    folds = []
    fold_id = 1
    start = 0
    while True:
        train_end = start + config.train_window
        val_end = train_end + config.val_window
        test_end = val_end + config.test_window
        if test_end > len(tuning_zone):
            break
        train = tuning_zone.iloc[start:train_end]
        val = tuning_zone.iloc[train_end:val_end]
        test = tuning_zone.iloc[val_end:test_end]
        fold = WalkForwardFold(
            fold_id=fold_id,
            train=train,
            val=val,
            test=test,
            train_start=train.index[0],
            train_end=train.index[-1],
            test_end=test.index[-1],
        )
        folds.append(fold)
        log.debug(str(fold))
        start += config.step_size
        fold_id += 1
    if not folds:
        raise BacktestError(
            'No valid folds generated',
            details={'tuning_rows': len(tuning_zone), 'min_required': min_required}
        )
    log.success(f'Generated {len(folds)} walk-forward folds')
    return folds

def get_holdout(df, config):
    holdout_size = int(len(df) * config.holdout_pct)
    holdout = df.iloc[-holdout_size:]
    log.info(
        f'Holdout set retrieved | '
        f'Rows: {len(holdout)} | '
        f'Period: {holdout.index[0].date()} to {holdout.index[-1].date()}'
    )
    return holdout

def get_feature_columns(df):
    exclude = {'target', 'forward_return', 'Open', 'High', 'Low', 'Volume'}
    features = [c for c in df.columns if c not in exclude]
    log.debug(f'Feature columns ({len(features)}): {features}')
    return features
