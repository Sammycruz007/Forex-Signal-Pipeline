import sys
from pathlib import Path
from loguru import logger


def setup_logger(log_level: str= "INFO", log_to_file : bool = True) -> logger:
    """
    Configure and return the application logger.

    Args:
        log_level (str): The logging level ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL").
        log_to_file (bool): Whether to persist logs  to a disk.

    Returns:
        Configured loguru logger instance.
    """
    #Remove default loguru handler
    logger.remove()

    # Console handler - human readable in development
    logger.add(
        sys.stdout, 
        level=log_level, 
        format=("<green>{time:YYYY-MM-DD HH:mm:ss}</green> "
            "| <level>{level: <8}</level> |" 
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> |"
             " <level>{message}</level>"
        ),
    colorize=True
    )   



    if log_to_file:
            # Ensure alogs directory exists
            Path("logs").mkdir(exist_ok=True)

            #File handler - structures, rotation-aware, persistent

            logger.add(
                "logs/pipeline_{time:YYYY-MM-DD}.log",
                level=log_level,
                format=("{time:YYYY-MM-DD HH:mm:ss} |"
                        "{level: <8}|"
                        "{name}:{function}:{line} |"
                        "{message}"
                ),
            rotation = " 1 day", # New file every day
            retention = "30 days", # Keep logs for 30 days
            compression = "zip" # Compress old logs
            enqueue = True # Async logging for performance
            )
    return logger

# Module-level logger instance - import this directly
log = setup_logger()
