from src.utils.logger import log
from src.utils.exception import (
     ForexPipelineError, 
     DataIngestionError, 
     DataValidationError, 
     DataProcessingError, 
     FeatureEngineeringError, 
     ModelTrainingError, 
     ModelLoadError, 
     ModelPredictionError
)
EOF

# This means in every future module, imports will look like this.