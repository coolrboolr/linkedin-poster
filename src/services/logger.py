import logging
import sys

def get_logger(name: str) -> logging.Logger:
    """
    Returns a configured logger with stdout handler.
    Prevents duplicate logging.
    """
    logger = logging.getLogger(name)
    logger.propagate = False
    
    # Check if a StreamHandler already exists
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
