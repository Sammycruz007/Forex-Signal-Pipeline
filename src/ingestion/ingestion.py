import yfinance as yf
import pandas as pd
from Pathlib import path
from datetime import datetime, time
from typing import Optional

from src.utils.logger import log
from src.utils.exception import (
    DataIngestionError,
    DataValidationError,
)
from src.utils.config import CONFIG

"""
This data ingestion module downloads the raw OHLCV data, valiadates
it and saves it to data/raw as Parquet files.
"""


# minimum numbers of rows we accept as valid data
MIN_REQUIRED_ROWS = 100

# Maximum percentage of nulls we tolerate per column
MAX_NULL_PCT = 0.05

#Required OHLCV columns
REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


# _____ Core functions _____

def download_ohlcv(
        ticker: str,
        period_years: int,
        interval: str,
        end_date: Optional[datetime]= None,
    )-> pd.DataFrame:

    """
    Download raw OHLCV data for a single forex pair from yfinance

    Args:
        ticker: yfinance ticker symbol e.g.  " EURUSD = X"
        period_years: How many years of history to fetch
        interval: Data frequency
        end_date:  End data for download(defaults to today)

    Returns:
        Raw OHLCV Dataframe with DatetimeIndex

    Raises:
        DataIngestionError: if download fails or return empty data
    """

    log.info(
        f"Downloading {ticker} |"
        f"From: {start.date()} To: {end.date()} |"
        f"Interval: {interval}"
    )

    try:
        df = yf.download(
            tickers = ticker,
            start = start.strftime("%Y-%m-%d"),
            end = end.strftime("%Y-%m-%d"),
            interval = interval,
            auto_adjust= True, #Adjust for splits and dividends
            progress= False, # Suppress yfinance progress bar in logs
        )

    except Exception as e:
        raise DataIngestionError(
            f"yfinance download failed for {ticker}",
            details= {"ticker": ticker, "error": str(e)}
        )
    
    # _____ Empty response check _____
    if df is None or df.empty:
        raise DataIngestionError(
            f"yfinance return empty data for {ticker}",
            details={
                "ticker": ticker
                "start": str(start.date()),
                "end": str(end.date()),
            }
        )
    
    log.success(f"Downloaded {len(df)} rows for {ticker}")
    return df


def validate_ohlcv(df: pd.DataFrame,ticker:str)-> pd.DataFrame:
    """
    Validates raw OHLCV data quality before saving
    
    Checks:
    - Required columns are present
    - Minimum row count is met
    - Null percentage is within tolerance
    - High ->Low
    - no duplicate index entries


    Args:
        df: Raw OHLCV DataFrame
        ticker: Ticker name for logging context

    Returns:
        Validated DataFrame

    Raises:
        DataValidationError: If data fails any quality check
    """

    log.info(f"Validating data {ticker} | shape: {df.shape}")


    # _____ Flatten MultiIndex columns if present ______
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        log.debug("Flattened MultiIndex  column from yfinance")

    # _____ Required columns _____
    missing_cols = [ c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise DataValidationError(
            f"Insufficient data for {ticker}",
            details= {
                "rows_found": len(df),
                "minimum_required": MIN_REQUIRED_ROWS
            }
        )
    
    # ____ Null Check _____
    null_pct = df[REQUIRED_COLUMNS].isnull().mean()
    violating = null_pct[null_pct > MAX_NULL_PCT]

    if not violating.empty:
        raise DataValidationError(
            f" Null values exceed {MAX_NULL_PCT * 100}% threshold in {ticker}",
            details= {"null_percentage": }
        )

    # _____ Price sanity check _____
    invalid_hl = df[df["High"]  < df["Low"]]
    if not invalid_hl.empty:
        raise DataValidationError(
            f"Found {len(invalid_hl)} row where High < Low in {ticker}",
            details= { "first_occurence": str(invalid_hl.index[0])}
        )
    
    # _____ Duplicate Index check _____
     duplicates = df.index.duplicates()
    if duplicates > 0:
        log.warning(f"Removing {duplicates} duplicates timestamps in {ticker}")
        df = df[~df.index.duplicated(keep = "last")]

    # _____ Drop any remaining nulls _____
    rows_before = len(df)
    df = df.dropna(subset=REQUIRED_COLUMNS)
    dropped = rows_before - len(df)

    if dropped > 0:
        log.warning(f"Dropped {dropped} rows with nulls in {ticker}")
    
    log.success(f"Validation passed for { ticker} | clean rows: {len(df)}")
    return df


def save_raw_data(df: pd.DataFrame, ticker: str, raw_path: str) -> Path:
    clean_ticker = ticker.replace("=X", "").replace("/", "")
    output_dr = Path(raw_path)
    output_dr.mkdir(parents =True, exist_ok = True)

    file_path = output_dr/ f"{clean_ticker}-raw.parquet"

    try:
        df.to_parquet(file_path, index = True)
        log.success(f"Saved raw data ->{file_path} | Rows: {len(df)}")

    except Exception as e:
        raise DataIngestionError(
            f"Failed to save raw data for {ticker}",
            details= {"path": str(file_path), "error": str(e)}
        )
    return file_path

def run_ingestion() -> dict:
    """
    Main entry point for thr ingestion pipeline.
    Downloads validates and saves file for all configured pairs
    """

    log.info("=" * 60)
    log.info("STARTING DATA INGESTION PIPELINE")
    log.info("=" * 60)

    # _____ Load config values _____
    tickers = CONFIG["data"]["pairs"]
    interval = CONFIG["data"]["interval"]
    period_years = CONFIG["data"]["period_years"]
    raw_path = CONFIG["data"]["raw_path"]

    log.info(f" Tickers: {tickers}")
    log.info(f" Interval: {interval} | period: {period_years}")
    

    results = {}
    failed = []


    # _____ Process each ticker _____
    for ticker in tickers:
        log.info(f"processing {ticker}...")
        
        try:
            # Step 1: Download
            raw_df = download_ohlcv(
                ticker = ticker
                period_years=  period_years
                interval = interval,
            )

            # Step 2
            clean_df = validate_ohlcv(
                df = raw_df,
                ticker = ticker,
            )

            # Step 3
            save_path = save_raw_data(
                df = clean_df,
                ticker = ticker,
                raw_path= raw_path
            )

            results[ticker] = {
                "status": "success",
                "rows" : len(clean_df),
                "path": str(saved_path),
                "date_range": (
                    f"{clean_df.index.min().date()} ->"
                    f"{clean_df.index.max().date()}"
                )

            }
        # Catch per-ticker failures - dont let one bad ticker kill others
        except (DataIngestionError, DataValidationError) as e:
            log.error(f" Failed to ingest {ticker}: {e}")
            results[ticker] = {"status": "failed", "error": str(e)}
            failed.append(ticker)

    # _____ Final Summary ____
    log.info("=" * 60)
    log.info("INGESTION SUMMARY")
    log.info("=" * 60)

    for ticker, result in results.items():
        if result["status"] == "success":
            log.success(
                f"✅ { ticker} |"
                f"{result["rows"]} rows |"
                f" {result["date_range"]}"
            )
        else:
            log.error(f"x {ticker} |{result["error"]}")
    
    # __ Fail hard only if ALL tickers failed
    if len(failed) == len(tickers):
        raise DataIngestionError(
            "All tickers failed ingestion. Pipeline cannot continue.",
            details={"failed_tickers": failed}
        )
    
    log.info("Data ingestion completed")
    return results

# _____ Script entry point _____
if __name__ == "__main__":
    run_ingestion()
EOF





