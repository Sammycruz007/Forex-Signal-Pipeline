import yfinance as yf
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
from src.utils.logger import log
from src.utils.exceptions import DataIngestionError, DataValidationError
from src.utils.config import CONFIG

MIN_REQUIRED_ROWS = 100
MAX_NULL_PCT = 0.05
REQUIRED_COLUMNS = ['Open', 'High', 'Low', 'Close', 'Volume']

def download_ohlcv(ticker, period_years, interval, end_date=None):
    end = end_date or datetime.today()
    start = end - timedelta(days=period_years * 365)
    log.info(f'Downloading {ticker} | From: {start.date()} To: {end.date()}')
    try:
        df = yf.download(
            tickers=ticker,
            start=start.strftime('%Y-%m-%d'),
            end=end.strftime('%Y-%m-%d'),
            interval=interval,
            auto_adjust=True,
            progress=False,
        )
    except Exception as e:
        raise DataIngestionError(
            f'yfinance download failed for {ticker}',
            details={'ticker': ticker, 'error': str(e)}
        )
    if df is None or df.empty:
        raise DataIngestionError(
            f'yfinance returned empty data for {ticker}',
            details={'ticker': ticker, 'start': str(start.date()), 'end': str(end.date())}
        )
    log.success(f'Downloaded {len(df)} rows for {ticker}')
    return df

def validate_ohlcv(df, ticker):
    log.info(f'Validating data for {ticker} | Shape: {df.shape}')
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise DataValidationError(f'Missing columns in {ticker}',
            details={'missing': missing_cols})
    if len(df) < MIN_REQUIRED_ROWS:
        raise DataValidationError(f'Insufficient data for {ticker}',
            details={'rows_found': len(df), 'minimum_required': MIN_REQUIRED_ROWS})
    null_pct = df[REQUIRED_COLUMNS].isnull().mean()
    violating = null_pct[null_pct > MAX_NULL_PCT]
    if not violating.empty:
        raise DataValidationError(f'Too many nulls in {ticker}',
            details={'null_percentages': violating.to_dict()})
    invalid_hl = df[df['High'] < df['Low']]
    if not invalid_hl.empty:
        raise DataValidationError(f'High < Low found in {ticker}',
            details={'first_occurrence': str(invalid_hl.index[0])})
    duplicates = df.index.duplicated().sum()
    if duplicates > 0:
        log.warning(f'Removing {duplicates} duplicate timestamps in {ticker}')
        df = df[~df.index.duplicated(keep='last')]
    rows_before = len(df)
    df = df.dropna(subset=REQUIRED_COLUMNS)
    dropped = rows_before - len(df)
    if dropped > 0:
        log.warning(f'Dropped {dropped} rows with nulls in {ticker}')
    log.success(f'Validation passed for {ticker} | Clean rows: {len(df)}')
    return df

def save_raw_data(df, ticker, raw_path):
    clean_ticker = ticker.replace('=X', '').replace('/', '')
    output_dir = Path(raw_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / f'{clean_ticker}_raw.parquet'
    try:
        df.to_parquet(file_path, index=True)
        log.success(f'Saved raw data to {file_path} | Rows: {len(df)}')
    except Exception as e:
        raise DataIngestionError(f'Failed to save raw data for {ticker}',
            details={'path': str(file_path), 'error': str(e)})
    return file_path

def run_ingestion():
    log.info('STARTING DATA INGESTION PIPELINE')
    tickers = CONFIG['data']['pairs']
    interval = CONFIG['data']['interval']
    period_years = CONFIG['data']['period_years']
    raw_path = CONFIG['data']['raw_path']
    log.info(f'Tickers: {tickers} | Interval: {interval} | Period: {period_years} years')
    results = {}
    failed = []
    for ticker in tickers:
        log.info(f'Processing {ticker}...')
        try:
            raw_df = download_ohlcv(
                ticker=ticker,
                period_years=period_years,
                interval=interval,
            )
            clean_df = validate_ohlcv(df=raw_df, ticker=ticker)
            saved_path = save_raw_data(
                df=clean_df,
                ticker=ticker,
                raw_path=raw_path,
            )
            results[ticker] = {
                'status': 'success',
                'rows': len(clean_df),
                'path': str(saved_path),
                'date_range': f'{clean_df.index.min().date()} to {clean_df.index.max().date()}',
            }
        except (DataIngestionError, DataValidationError) as e:
            log.error(f'Failed to ingest {ticker}: {e}')
            results[ticker] = {'status': 'failed', 'error': str(e)}
            failed.append(ticker)
    log.info('INGESTION SUMMARY')
    for ticker, result in results.items():
        if result['status'] == 'success':
            log.success(f"OK {ticker} | {result['rows']} rows | {result['date_range']}")
        else:
            log.error(f"FAILED {ticker} | {result['error']}")
    if len(failed) == len(tickers):
        raise DataIngestionError('All tickers failed.', details={'failed': failed})
    log.info('Data ingestion complete.')
    return results

if __name__ == '__main__':
    run_ingestion()
