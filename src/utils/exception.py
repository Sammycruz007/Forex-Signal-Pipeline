# Base Exception

class ForexPipelineError(Exception):
    """
    Base exception for all pipeline errors. Catch this to handlemy pipeline failure generically.
    """

    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.message= message
        self.details = details or {}

    def __str__(self):
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message
    
# Data Layer Exception
class DataIngestionError(ForexPipelineError):
    """
    Raised when OHLCV data cannot be fetched from the data source.
    Examples: network timeout, invalid ticker, API rate limit
    """
    pass


class DataValidationError(ForexPipelineError):
    """
    Raised when fetched data  fails quality checks.
    Examples: missing columns, too many nulls, zero-length dataframe
    """
    pass

class DataProcessingError(ForexPipelineError):
     """
    Raised when data transformer or cleaning fails.
    Examples: resampling error, type conversion failure
    """
pass

# Feature Engineering Exceptions
class FeatureEngineeringError(ForexPipelineError):
    """
    Raised when technical indicator computation fails fail.
    Examples: Insufficient data for indicator period, NaN-only output
    """
    pass

#Model Exceptions
class ModelTrainingError(ForexPipelineError):
    """
    Raised when model training fails due to issues in the training loop or data.
    Examples: LSTM convergence failure, invalid hyperparameters, prophet fitting error
    """
    pass

class ModelLoadError(ForexPipelineError):
    """
    Raised when a pre-trained (saved) model cannot be loaded from disk or MLflow
    Examples: file not found, incompatible model version, missing artifacts in MLflow
    """
    pass

class ModelPredictionError(ForexPipelineError):
    """
    Raised when model prediction fails due to issues in the input data or model state.
    Examples: input data shape mismatch, model not loaded, runtime error during prediction
    """
    pass

# Backtesting Exceptions
class BacktestingError(ForexPipelineError):
    """
    Raised when backtesting fails due to issues in the backtesting logic or data.
    Examples: insufficient historical data, invalid strategy parameters, runtime error during backtest
    """
    pass

class SharpeThresholdError(ForexPipelineError):
    """
    Raised when the model's Sharpe ratio fails the Sharpe ratio gate during backtesting.
    GitHub Action catches this to fail the CI/CD pipeline
    Examples: model underperformance, excessive volatility in returns
    """
    pass


class PipelineConfigError(ForexPipelineError):
    """
    Raised when config.yaml is missing required fields or has invalid values.
    """
    pass



