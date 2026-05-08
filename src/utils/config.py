from pathlib import Path
import yaml
from src.utils.exception import PipelineConfigError
from src.utils.logger import log


"""
Configuration loader for the entire pipeline. All parameters comes from config.yaml file.
No hardcoded values anywhere else in the codebase.
"""

def load_config(config_path: str = "configs/config.yaml")-> dict:
    """
    Load and validate the pipeline configuration file.

    Args:
        config_path: Path to the yaml configuration file.

    Return:
        Dictionary of configuration parameters

    Raises:
        PipelineConfigError: If file is missing or required keys are absent

    """


path = Path(config_path)

# _____ Check thaat file Exist ______
if not path.exist():
    raise PipelineConfigError(
        f"Config file not found at:{config_path}",
        details ={"expected_path": str(path.absolute())}
    )

# _____ Load YAML _____
try:
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    log.info(f"Configuration loaded from {config_path}")

except yaml.YAMLError as e:
    raise PipelineConfigError(
        "Failed to parse config.yaml - check for syntax errors",
        details = {"error": str(e)}
    )

#_____ Validate required top-level keys ________
 required_keys = ["data", "features", "models", "backtesting", "mlfow"]
missing = [key for key in required_keys if key not in config]

if missing:
    raise PipelineConfigError(
        " Config file is mising required sections",
        details = {"missing_keys": missing}
    )

log.debug(f"Config validation passed. Sections found: {list(config.keys())}")
return config


# Module-level config instance - import this directly
 CONFIG = load_config

EOF