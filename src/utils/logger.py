# src/utils/logger.py

import logging
from pathlib import Path
import datetime
from typing import Optional

# Define the name for the root application logger.
_APP_ROOT_LOGGER_NAME = 'ADAPT_EEG_Application'

def configure_app_logger(log_dir: str = 'results/logs', 
                         log_file_name: Optional[str] = None, 
                         log_level: int = logging.INFO) -> None:
    """
    Configures the root application logger. This function should be called ONCE
    at the application's entry point (e.g., in main.py) to set up global
    logging handlers and formatter. Subsequent calls will NOT duplicate handlers.

    Logs will be directed to both the console and a specified file.

    Args:
        log_dir (str): Directory where log files will be stored.
        log_file_name (str, optional): Name of the log file. If None, a timestamp-based
                                       name will be generated (e.g., 'experiment_YYYYMMDD_HHMMSS.log').
        log_level (int): The minimum logging level for the application (e.g., logging.INFO,
                         logging.DEBUG, logging.WARNING).
    """
    # Ensure log directory exists
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    if log_file_name is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_name = f"experiment_{timestamp}.log"
    
    log_file_path = log_path / log_file_name

    # Get the root logger instance for the application.
    # Using `_APP_ROOT_LOGGER_NAME` ensures we always configure the same global logger.
    logger = logging.getLogger(_APP_ROOT_LOGGER_NAME)
    logger.setLevel(log_level)
    # Prevent propagation to the Python's default root logger. 
    # This prevents log messages from being handled multiple times if the default root
    # logger also has handlers configured.
    logger.propagate = False 

    # Add handlers only if they haven't been added yet.
    # This design prevents duplicate log messages if `configure_app_logger` is called
    # multiple times in the application's lifecycle.
    if not logger.handlers:
        # Define a common formatter for all handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Console handler: Outputs log messages to stdout/stderr
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler: Writes log messages to a specified file
        file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Root application logger '{_APP_ROOT_LOGGER_NAME}' has been configured.")
        logger.info(f"All application logs will be redirected to: {log_file_path}")
    else:
        logger.debug(f"Root application logger '{_APP_ROOT_LOGGER_NAME}' already configured. Skipping handler setup.")
        # In this scenario, existing handlers remain. If `log_file_name` were to change
        # in a subsequent call, existing file handlers would need to be reconfigured/removed.
        # For typical single-time application setup, this is sufficient.

def get_logger(name: str) -> logging.Logger:
    """
    Retrieves a logger instance by a given name. This is the primary function for
    individual modules to obtain a logger.
    
    It leverages Python's hierarchical logging system. If `configure_app_logger` has
    been called (which should be done at application startup), any logger obtained
    via this function will automatically inherit the handlers and logging level
    from the root application logger (`_APP_ROOT_LOGGER_NAME`).

    Args:
        name (str): The name of the logger to retrieve. For module-specific
                    logging, it's highly recommended to pass `__name__` here
                    (e.g., `logger = get_logger(__name__)`). This creates a
                    hierarchical logger name (e.g., 'ADAPT_EEG_Application.src.module_name').

    Returns:
        logging.Logger: The specific logger instance for the given name.
    """
    # If the requested name is the root application logger's name, return it directly.
    # Otherwise, prepend the root logger's name to create a hierarchical logger.
    # This ensures all module-level logs are nested under our main application logger.
    if name == _APP_ROOT_LOGGER_NAME:
        return logging.getLogger(_APP_ROOT_LOGGER_NAME)
    else:
        return logging.getLogger(f"{_APP_ROOT_LOGGER_NAME}.{name}")

