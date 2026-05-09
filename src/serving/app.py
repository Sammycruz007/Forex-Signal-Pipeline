import os
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from src.utils.logger import log
from src.utils.config import CONFIG
from src.utils.exceptions import ModelLoadError, ModelPredictionError

app = FastAPI(
    title='Forex Signal Pipeline API',
    description='Serves BUY/SELL/HOLD signals for forex pairs',
    version='1.0.0',
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)

LOADED_MODELS = {}

class PredictionResponse(BaseModel):
    ticker: str
    signal: str
    probability: float
    direction: str
    timestamp: str
    model_version: str

class HealthResponse(BaseModel):
    status: str
    models_loaded: list
    timestamp: str

class MetricsResponse(BaseModel):
    ticker: str
    sharpe_ratio: float
    max_drawdown: float
    total_return: float
    win_rate: float

def load_model(ticker):
    clean_ticker = ticker.replace('=X', '').replace('/', '')
    model_path = Path(f'models/ensemble/{clean_ticker}_ensemble.pkl')
    if not model_path.exists():
        raise ModelLoadError(
            f'No trained model found for {ticker}',
            details={'path': str(model_path)}
        )
    model = joblib.load(model_path)
    log.success(f'Model loaded for {ticker} from {model_path}')
    return model

@app.on_event('startup')
async def startup_event():
    log.info('FastAPI starting up — loading models...')
    tickers = CONFIG['data']['pairs']
    for ticker in tickers:
        try:
            LOADED_MODELS[ticker] = load_model(ticker)
            log.success(f'Loaded model for {ticker}')
        except ModelLoadError as e:
            log.warning(f'Could not load model for {ticker}: {e}')
    log.info(f'Startup complete. Models loaded: {list(LOADED_MODELS.keys())}')

@app.get('/health', response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status='healthy',
        models_loaded=list(LOADED_MODELS.keys()),
        timestamp=datetime.utcnow().isoformat(),
    )

@app.get('/predict/{ticker}', response_model=PredictionResponse)
async def predict(ticker: str):
    log.info(f'Prediction request for {ticker}')
    if ticker not in LOADED_MODELS:
        try:
            LOADED_MODELS[ticker] = load_model(ticker)
        except ModelLoadError:
            raise HTTPException(
                status_code=404,
                detail=f'No trained model available for {ticker}. Train the model first.'
            )
    try:
        import yfinance as yf
        raw_df = yf.download(
            tickers=ticker,
            period='6mo',
            interval='1d',
            auto_adjust=True,
            progress=False,
        )
        if raw_df.empty:
            raise HTTPException(
                status_code=503,
                detail=f'Could not fetch live data for {ticker}'
            )
        if isinstance(raw_df.columns, pd.MultiIndex):
            raw_df.columns = raw_df.columns.get_level_values(0)
        from src.features.engineer import build_features
        feature_df = build_features(raw_df.copy(), ticker)
        model = LOADED_MODELS[ticker]
        signal, probability = model.predict(feature_df)
        direction = 'UP' if signal == 'BUY' else 'DOWN' if signal == 'SELL' else 'NEUTRAL'
        log.success(f'Prediction for {ticker}: {signal} ({probability:.3f})')
        return PredictionResponse(
            ticker=ticker,
            signal=signal,
            probability=round(probability, 4),
            direction=direction,
            timestamp=datetime.utcnow().isoformat(),
            model_version='1.0.0',
        )
    except Exception as e:
        log.error(f'Prediction failed for {ticker}: {e}')
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/metrics/{ticker}', response_model=MetricsResponse)
async def get_metrics(ticker: str):
    import json
    clean_ticker = ticker.replace('=X', '').replace('/', '')
    results_path = Path(f'logs/backtest_{clean_ticker}.json')
    if not results_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f'No backtest results found for {ticker}'
        )
    with open(results_path) as f:
        data = json.load(f)
    return MetricsResponse(
        ticker=ticker,
        sharpe_ratio=data['sharpe_ratio'],
        max_drawdown=data['max_drawdown'],
        total_return=data['total_return'],
        win_rate=data['win_rate'],
    )

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(
        'src.serving.app:app',
        host=CONFIG['serving']['host'],
        port=CONFIG['serving']['port'],
        reload=True,
    )
