from loguru import logger as loguru_logger
from loguru._logger import Logger
import sys
from datetime import datetime
import os

def log_file_path():
    base_dir = './logs'
    month = datetime.now().strftime('%Y-%m')
    day = datetime.now().strftime('%d')

    month_dir = os.path.join(base_dir, month)
    

    log_dir = os.path.join(month_dir, f'{day}.log')
    os.makedirs(log_dir, exist_ok=True)

    return log_dir


def logger_config(logger: Logger)-> Logger:

    # Remove loggers criados anteriormente
    logger.remove()
    logger_dir = log_file_path()

    # Default logger que escreve no stdout
    logger.add(
        sys.stdout, 
        level="DEBUG",
        colorize=True,
    )

    logger.add(
        logger_dir,
        level="DEBUG",
        rotation="00:00",
        retention="7 days",
        enqueue=True,
        backtrace=True,
        colorize=False,
        
    )

    return logger

logger = logger_config(loguru_logger)
  