# FOREX SIGNAL PIPELINE
End-to-end MLOps pipeline for algorithmic Forex trading signals

## Stack
-**Models**: Prophet + LSTM +Scikit-learn Ensemble
-**Tracking**: MLFlow
-**Versioning**: DVC + Git
-**Serving**: FastAPI + Docker
-**Orchestration**: Apache Airflow 
-**Dashboard**: Streamlit
-**CI/CD**: GitHub Actions ( Sharpe Ration Gate)

## Architecture
Raw OHLCV > Feature Engineering > Ensemble Model > BackTest gate > FastAPI > Streamlit

## Quick Start
...
    bash
docker-compose up

## CI/CD
No model is promoted to production without achieving a Sharpe Ratio >= 1.0 on a 6-month out-of-sample backtest.
EOF
