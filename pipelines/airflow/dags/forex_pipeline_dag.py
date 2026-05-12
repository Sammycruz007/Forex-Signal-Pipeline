from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
from airflow.utils.dates import days_ago

default_args = {
    'owner':            'forex-pipeline',
    'depends_on_past':  False,
    'email_on_failure': False,
    'email_on_retry':   False,
    'retries':          1,
    'retry_delay':      timedelta(minutes=5),
}

with DAG(
    dag_id             = 'forex_signal_pipeline',
    default_args       = default_args,
    description        = 'Daily forex signal generation pipeline',
    schedule_interval  = '0 6 * * 1-5',
    start_date         = days_ago(1),
    catchup            = False,
    tags               = ['forex', 'mlops'],
) as dag:

    ingest = BashOperator(
        task_id         = 'data_ingestion',
        bash_command    = 'cd /app && python -m src.ingestion.ingest',
    )

    engineer = BashOperator(
        task_id         = 'feature_engineering',
        bash_command    = 'cd /app && python -m src.features.engineer',
    )

    backtest = BashOperator(
        task_id         = 'run_backtest',
        bash_command    = 'cd /app && python -m src.backtesting.backtest',
    )

    serve = BashOperator(
        task_id         = 'restart_api',
        bash_command    = 'echo "Signal pipeline complete. API serving latest model."',
    )

    ingest >> engineer >> backtest >> serve
