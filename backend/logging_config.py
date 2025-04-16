import logging
import os

class LoggerConfig:
    LOG_DIR = "logs_gunicorn"
    os.makedirs(LOG_DIR, exist_ok=True)  # Ensure log directory exists

    LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    
    @staticmethod
    def get_logger(name):
        """Returns a logger instance with a unified format and handlers."""
        logger = logging.getLogger(name)
        logger.handlers = []
        
        logger.setLevel(logging.DEBUG)  # Capture all logs (INFO, DEBUG, ERROR)
        
        # Create a formatter with a proper timestamp format
        formatter = logging.Formatter(LoggerConfig.LOG_FORMAT, LoggerConfig.DATE_FORMAT)
        # Handlers
        app_log_handler = logging.FileHandler(os.path.join(LoggerConfig.LOG_DIR, "app.log"))
        app_log_handler.setFormatter(formatter)
        app_log_handler.setLevel(logging.DEBUG)  # General logs

        # Add handlers
        logger.addHandler(app_log_handler)
        
        logger.propagate = False

        return logger
