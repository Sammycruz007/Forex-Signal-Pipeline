import sys
from pathlib import Path
from loguru import logger

def setup_logger(log_level='INFO', log_to_file=True):
    logger.remove()
    logger.add(
        sys.stdout,
        level=log_level,
        format=(
            '<green>{time:YYYY-MM-DD HH:mm:ss}</green> | '
            '<level>{level: <8}</level> | '
            '<cyan>{name}</cyan>:<cyan>{line}</cyan> | '
            '<level>{message}</level>'
        ),
        colorize=True,
    )
    if log_to_file:
        Path('logs').mkdir(exist_ok=True)
        logger.add(
            'logs/pipeline_{time:YYYY-MM-DD}.log',
            level=log_level,
            format='{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} | {message}',
            rotation='1 day',
            retention='30 days',
            compression='zip',
            enqueue=True,
        )
    return logger

log = setup_logger()
