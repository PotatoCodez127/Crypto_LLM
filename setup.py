"""
Setup script for the trading system.
"""

import os
import sys
from config.settings import DATA_STORAGE_PATH


def setup_environment():
    """Setup the trading environment and directories."""
    print("Setting up trading environment...")

    # Create data directory
    data_dir = os.path.dirname(DATA_STORAGE_PATH)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created data directory: {data_dir}")

    # Create logs directory
    logs_dir = "./logs"
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
        print(f"Created logs directory: {logs_dir}")

    print("Environment setup complete!")


if __name__ == "__main__":
    setup_environment()
