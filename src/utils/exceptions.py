class ForexPipelineError(Exception):
    def __init__(self, message, details=None):
        super().__init__(message)
        self.message = message
        self.details = details or {}
    def __str__(self):
        if self.details:
            return f'{self.message} | Details: {self.details}'
        return self.message

class DataIngestionError(ForexPipelineError): pass
class DataValidationError(ForexPipelineError): pass
class DataProcessingError(ForexPipelineError): pass
class FeatureEngineeringError(ForexPipelineError): pass
class ModelTrainingError(ForexPipelineError): pass
class ModelLoadError(ForexPipelineError): pass
class ModelPredictionError(ForexPipelineError): pass
class BacktestError(ForexPipelineError): pass
class SharpeThresholdError(ForexPipelineError): pass
class PipelineConfigError(ForexPipelineError): pass
