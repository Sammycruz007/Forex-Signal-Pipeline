from pathlib import Path
import yaml
from src.utils.logger import log
from src.utils.exceptions import PipelineConfigError

def load_config(config_path='configs/config.yaml'):
    path = Path(config_path)
    if not path.exists():
        raise PipelineConfigError(
            f'Config file not found at: {config_path}',
            details={'expected_path': str(path.absolute())}
        )
    try:
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        log.info(f'Configuration loaded from {config_path}')
    except yaml.YAMLError as e:
        raise PipelineConfigError('Failed to parse config.yaml',
            details={'error': str(e)})
    required_keys = ['data', 'features', 'models', 'backtesting', 'mlflow']
    missing = [key for key in required_keys if key not in config]
    if missing:
        raise PipelineConfigError('Config missing required sections',
            details={'missing_keys': missing})
    log.debug(f'Config OK. Sections: {list(config.keys())}')
    return config

CONFIG = load_config()
