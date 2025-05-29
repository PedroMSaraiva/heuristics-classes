from loguru import logger
import sys

def get_logger(script_name=None):
    logger.remove()
    fmt = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )
    logger.add(sys.stderr, format=fmt, colorize=True, level="INFO")
    if script_name:
        logger.add(f"logs/{script_name}_{{time:YYYYMMDD_HHmmss}}.log", format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}", level="DEBUG")
    else:
        logger.add("logs/project_{time:YYYYMMDD_HHmmss}.log", format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}", level="DEBUG")
    return logger 