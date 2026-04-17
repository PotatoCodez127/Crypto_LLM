"""
Logging utility for the trading system.
"""

import logging
import os
from datetime import datetime


class TradingLogger:
    """Custom logger for trading system activities."""

    def __init__(self, name="trading_ai", log_file="./logs/trading_system.log"):
        """Initialize logger with file and console handlers."""
        # Create logs directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        # Prevent duplicate handlers
        if not self.logger.handlers:
            # Create file handler
            file_handler = logging.FileHandler(log_file)
            file_formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)

            # Create console handler
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                "%(asctime)s - %(levelname)s - %(message)s"
            )
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)

    def info(self, message):
        """Log info message."""
        self.logger.info(message)

    def warning(self, message):
        """Log warning message."""
        self.logger.warning(message)

    def error(self, message):
        """Log error message."""
        self.logger.error(message)

    def debug(self, message):
        """Log debug message."""
        self.logger.debug(message)


# Global logger instance
trading_logger = TradingLogger()

if __name__ == "__main__":
    # Test the logger
    logger = TradingLogger()
    logger.info("Logger initialized successfully")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
